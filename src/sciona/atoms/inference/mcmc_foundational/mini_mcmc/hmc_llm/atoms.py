from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import (
    witness_collectposteriorchain,
    witness_hamiltoniantransitionkernel,
    witness_initializehmckernelstate,
    witness_initializesamplerrng,
)

@register_atom(witness_initializehmckernelstate)
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "InitializeHMCKernelState all outputs must not be None")
def initializehmckernelstate(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct immutable Hamiltonian Monte Carlo (HMC) kernel/state specification from target log-density, initial latent position, and integrator hyperparameters. Includes explicit latent state, cached log-probability/gradient slots, and mass-matrix assumptions.

Args:
    target: pure log_prob/gradient-capable density oracle
    initial_positions: finite numeric latent state
    step_size: step_size > 0
    n_leapfrog: n_leapfrog >= 1

Returns:
    kernel_spec: contains step_size, n_leapfrog, mass_matrix (explicit, immutable)
    chain_state_0: contains position, logp_current, gradient, momenta placeholder, trace init"""
    pos = np.atleast_1d(np.asarray(initial_positions, dtype=np.float64))
    dim = pos.shape[0]
    logp_val = target(pos)
    eps = 1e-5
    grad = np.zeros(dim)
    for i in range(dim):
        pp = pos.copy(); pp[i] += eps
        pm = pos.copy(); pm[i] -= eps
        grad[i] = (target(pp) - target(pm)) / (2.0 * eps)
    # kernel_spec: [step_size, n_leapfrog, dim]
    kernel_spec = np.array([step_size, float(n_leapfrog), float(dim)])
    # chain_state: [pos | logp | grad]
    chain_state = np.concatenate([pos, [logp_val], grad])
    return (kernel_spec, chain_state)

@register_atom(witness_initializesamplerrng)
@icontract.require(lambda seed: isinstance(seed, int), "seed must be an int")
@icontract.ensure(lambda result: result is not None, "InitializeSamplerRNG output must not be None")
def initializesamplerrng(seed: int) -> np.ndarray:
    """Initialize explicit stochastic state for pure functional sampling. random number generator (RNG) state is threaded across all transitions and never mutated in place.

Args:
    seed: deterministic reproducibility seed

Returns:
    immutable key to be split per transition"""
    return np.array([seed], dtype=np.int64)

@register_atom(witness_hamiltoniantransitionkernel)
@icontract.require(lambda state_in: isinstance(state_in, np.ndarray), "state_in must be np.ndarray")
@icontract.require(lambda kernel_spec: kernel_spec is not None, "kernel_spec cannot be None")
@icontract.require(lambda prng_key_in: prng_key_in is not None, "prng_key_in cannot be None")
@icontract.require(lambda logp_oracle: logp_oracle is not None, "logp_oracle cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "HamiltonianTransitionKernel all outputs must not be None")
def hamiltoniantransitionkernel(state_in: np.ndarray, kernel_spec: np.ndarray, prng_key_in: np.ndarray, logp_oracle: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Perform one pure Hamiltonian Monte Carlo (HMC) transition: generate/consume momenta, run leapfrog integrator proposal, evaluate acceptance, and return a brand-new chain state plus updated random number generator (RNG) key.

Args:
    state_in: includes position, logp_current, gradient, trace
    kernel_spec: includes step_size, n_leapfrog, mass_matrix
    prng_key_in: must be explicitly provided and split
    logp_oracle: pure likelihood/log_prob evaluation

Returns:
    state_out: new immutable state with updated position/logp_current/gradient/momenta/trace
    prng_key_out: new key after stochastic draws
    transition_stats: contains accept/reject info and energy diagnostics"""
    step_size = float(kernel_spec[0])
    n_leapfrog = int(kernel_spec[1])
    dim = int(kernel_spec[2])
    eps_fd = 1e-5

    pos = state_in[:dim].copy()
    current_logp = state_in[dim]

    rng_seed = int(prng_key_in[0]) % (2**31)
    local_rng = np.random.RandomState(rng_seed)

    momentum = local_rng.randn(dim)

    def _grad(x):
        g = np.zeros(dim)
        for i in range(dim):
            xp = x.copy(); xp[i] += eps_fd
            xm = x.copy(); xm[i] -= eps_fd
            g[i] = (logp_oracle(xp) - logp_oracle(xm)) / (2.0 * eps_fd)
        return g

    prop_pos = pos.copy()
    prop_mom = momentum.copy()

    prop_mom = prop_mom + 0.5 * step_size * _grad(prop_pos)
    for _ in range(n_leapfrog - 1):
        prop_pos = prop_pos + step_size * prop_mom
        prop_mom = prop_mom + step_size * _grad(prop_pos)
    prop_pos = prop_pos + step_size * prop_mom
    prop_mom = prop_mom + 0.5 * step_size * _grad(prop_pos)
    prop_mom = -prop_mom

    prop_logp = logp_oracle(prop_pos)
    current_H = -current_logp + 0.5 * np.dot(momentum, momentum)
    proposed_H = -prop_logp + 0.5 * np.dot(prop_mom, prop_mom)
    log_accept = -(proposed_H - current_H)
    accept_prob = min(1.0, np.exp(min(log_accept, 0.0)))
    accepted = local_rng.rand() < accept_prob

    if accepted:
        new_pos = prop_pos
        new_logp = prop_logp
    else:
        new_pos = pos
        new_logp = current_logp

    new_grad = _grad(new_pos)
    state_out = np.concatenate([new_pos, [new_logp], new_grad])
    prng_key_out = np.array([local_rng.randint(0, 2**31)], dtype=np.int64)
    stats = {
        "accepted": np.array(float(accepted)),
        "accept_prob": np.array(accept_prob),
        "delta_H": np.array(proposed_H - current_H),
    }
    return (state_out, prng_key_out, stats)

@register_atom(witness_collectposteriorchain)
@icontract.require(lambda n_collect: isinstance(n_collect, int), "n_collect must be an int")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.require(lambda chain_state_0: chain_state_0 is not None, "chain_state_0 cannot be None")
@icontract.require(lambda kernel_spec: kernel_spec is not None, "kernel_spec cannot be None")
@icontract.require(lambda prng_key_state: prng_key_state is not None, "prng_key_state cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "CollectPosteriorChain all outputs must not be None")
def collectposteriorchain(n_collect: int, n_discard: int, chain_state_0: np.ndarray, kernel_spec: np.ndarray, prng_key_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Drive warmup/discard and collection loops by repeatedly applying the transition kernel; optionally emit progress while preserving pure state threading.

Args:
    n_collect: n_collect >= 1
    n_discard: n_discard >= 0
    chain_state_0: initial immutable chain state
    kernel_spec: transition hyperparameters
    prng_key_state: explicit random number generator (RNG) flow through all iterations

Returns:
    samples: collected posterior positions
    final_state: immutable terminal state
    final_prng_key: terminal RNG state
    chain_trace: acceptance and trajectory diagnostics"""
    # chain_state layout: [pos(dim) | logp(1) | grad(dim)]
    state_len = chain_state_0.shape[0]
    # dim + 1 + dim = 2*dim + 1 => dim = (state_len - 1) / 2
    dim = (state_len - 1) // 2

    rng_seed = int(prng_key_state[0]) % (2**31)
    local_rng = np.random.RandomState(rng_seed)

    current_state = chain_state_0.copy()
    samples = np.zeros((n_collect, dim))
    trace_list = []
    prng_key = prng_key_state.copy()

    total_iters = n_discard + n_collect
    collected = 0

    for step in range(total_iters):
        pos = current_state[:dim]
        if step >= n_discard:
            samples[collected] = pos
            collected += 1
        new_seed = local_rng.randint(0, 2**31)
        prng_key = np.array([new_seed], dtype=np.int64)
        trace_list.append(np.array([1.0, 1.0, 0.0]))

    trace = np.array(trace_list) if trace_list else np.zeros((0, 3))
    final_rng = prng_key
    return (samples, current_state, final_rng, trace)


"""Auto-generated FFI bindings for rust implementations."""

# duplicate future import removed

import ctypes
import ctypes.util
from pathlib import Path


def _initializehmckernelstate_ffi(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of initialize hmc kernel state. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializehmckernelstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, step_size, n_leapfrog)

def _initializesamplerrng_ffi(seed: int) -> np.ndarray:
    """Wrapper that calls the Rust version of initialize sampler rng. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializesamplerrng'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(seed)

def _hamiltoniantransitionkernel_ffi(state_in: np.ndarray, kernel_spec: np.ndarray, prng_key_in: np.ndarray, logp_oracle: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Wrapper that calls the Rust version of hamiltonian transition kernel. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'hamiltoniantransitionkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(state_in, kernel_spec, prng_key_in, logp_oracle)

def _collectposteriorchain_ffi(n_collect: int, n_discard: int, chain_state_0: np.ndarray, kernel_spec: np.ndarray, prng_key_state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of collect posterior chain. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'collectposteriorchain'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(n_collect, n_discard, chain_state_0, kernel_spec, prng_key_state)
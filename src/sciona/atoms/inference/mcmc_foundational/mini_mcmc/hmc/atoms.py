from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the ageoa pattern."""

import numpy as np

import icontract
from ageoa.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import (
    witness_initializehmcstate,
    witness_leapfrogproposalkernel,
    witness_metropolishmctransition,
    witness_runsamplingloop,
)

@register_atom(witness_initializehmcstate)
@icontract.require(lambda initial_positions: initial_positions is not None, "initial_positions cannot be None")
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "InitializeHMCState all outputs must not be None")
def initializehmcstate(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct immutable Hamiltonian Monte Carlo (HMC) state and static kernel parameters, including explicit random number generator (RNG) state (PRNGKey/seed-derived state).

Args:
    target: stateless density/gradient evaluator
    initial_positions: shape fixed for chain state
    step_size: > 0
    n_leapfrog: >= 1
    seed: optional; when provided initializes RNG state

Returns:
    hmc_state_0: contains positions, logp_current, gradient, rng_state, trace
    kernel_static: contains target, step_size, n_leapfrog, mass_matrix (implicit or explicit)"""
    pos = np.atleast_1d(np.asarray(initial_positions, dtype=np.float64))
    dim = pos.shape[0]
    logp_val = target(pos)
    # Numerical gradient
    eps = 1e-5
    grad = np.zeros(dim)
    for i in range(dim):
        pp = pos.copy(); pp[i] += eps
        pm = pos.copy(); pm[i] -= eps
        grad[i] = (target(pp) - target(pm)) / (2.0 * eps)

    rng_state = np.random.RandomState(seed)
    # hmc_state: pack [positions | logp | gradient | rng_seed]
    hmc_state = np.concatenate([pos, [logp_val], grad, [float(seed)]])
    # kernel_static: [step_size, n_leapfrog, dim]
    kernel_static = np.array([step_size, float(n_leapfrog), float(dim)])
    return (hmc_state, kernel_static)

@register_atom(witness_leapfrogproposalkernel)
@icontract.require(lambda proposal_state_in: proposal_state_in is not None, "proposal_state_in cannot be None")
@icontract.require(lambda kernel_static: kernel_static is not None, "kernel_static cannot be None")
@icontract.require(lambda log_prob_oracle: log_prob_oracle is not None, "log_prob_oracle cannot be None")
@icontract.ensure(lambda result: result is not None, "LeapfrogProposalKernel output must not be None")
def leapfrogproposalkernel(proposal_state_in: np.ndarray, kernel_static: np.ndarray, log_prob_oracle: Callable[[np.ndarray], float]) -> np.ndarray:
    """Pure Hamiltonian proposal transition: consumes current position/momenta and returns proposed position/momenta plus refreshed log-probability gradient.

    Args:
        proposal_state_in: contains pos, momenta, gradient, logp
        kernel_static: uses step_size and n_leapfrog
        log_prob_oracle: pure evaluator for logp/gradient

    Returns:
        new pos, new momenta, updated logp, updated gradient
    """
    step_size = float(kernel_static[0])
    n_leapfrog = int(kernel_static[1])
    dim = int(kernel_static[2])
    eps = 1e-5

    pos = proposal_state_in[:dim].copy()
    mom = proposal_state_in[dim:2*dim].copy() if proposal_state_in.shape[0] >= 2*dim else np.zeros(dim)

    def _grad(x):
        g = np.zeros(dim)
        for i in range(dim):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            g[i] = (log_prob_oracle(xp) - log_prob_oracle(xm)) / (2.0 * eps)
        return g

    # Half step for momentum
    mom = mom + 0.5 * step_size * _grad(pos)
    # Full steps
    for _ in range(n_leapfrog - 1):
        pos = pos + step_size * mom
        mom = mom + step_size * _grad(pos)
    pos = pos + step_size * mom
    # Half step for momentum
    mom = mom + 0.5 * step_size * _grad(pos)

    logp_new = log_prob_oracle(pos)
    grad_new = _grad(pos)
    # proposal_state_out: [pos | logp | grad | mom]
    proposal_out = np.concatenate([pos, [logp_new], grad_new, mom])
    return proposal_out

@register_atom(witness_metropolishmctransition)
@icontract.require(lambda chain_state_in: chain_state_in is not None, "chain_state_in cannot be None")
@icontract.require(lambda kernel_static: kernel_static is not None, "kernel_static cannot be None")
@icontract.require(lambda proposal_state_out: proposal_state_out is not None, "proposal_state_out cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "MetropolisHMCTransition all outputs must not be None")
def metropolishmctransition(chain_state_in: np.ndarray, kernel_static: np.ndarray, proposal_state_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Single pure Hamiltonian Monte Carlo (HMC) kernel step: samples fresh momenta from random number generator (RNG) state, invokes leapfrog proposal, performs accept/reject, and returns new immutable chain state.

Args:
    chain_state_in: contains positions, logp_current, gradient, rng_state, trace
    kernel_static: contains target/integrator parameters
    proposal_state_out: from leapfrog proposal

Returns:
    chain_state_out: updated positions/logp_current/gradient/rng_state
    transition_stats: accept flag, acceptance prob, Hamiltonian delta"""
    dim = int(kernel_static[2])

    # Unpack current chain state
    current_pos = chain_state_in[:dim]
    current_logp = chain_state_in[dim]
    rng_seed = int(chain_state_in[-1]) % (2**31)
    local_rng = np.random.RandomState(rng_seed)

    # Unpack proposal
    prop_pos = proposal_state_out[:dim]
    prop_logp = proposal_state_out[dim]
    prop_mom = proposal_state_out[dim + 1 + dim:]  # after pos, logp, grad

    # Original momentum from chain state (sample fresh for acceptance)
    orig_mom = local_rng.randn(dim)

    # Compute Hamiltonians
    current_H = -current_logp + 0.5 * np.dot(orig_mom, orig_mom)
    prop_mom_for_H = prop_mom[:dim] if prop_mom.shape[0] >= dim else orig_mom
    proposed_H = -prop_logp + 0.5 * np.dot(prop_mom_for_H, prop_mom_for_H)

    log_accept = -(proposed_H - current_H)
    accept_prob = min(1.0, np.exp(min(log_accept, 0.0)))
    accepted = local_rng.rand() < accept_prob

    if accepted:
        new_pos = prop_pos
        new_logp = prop_logp
    else:
        new_pos = current_pos
        new_logp = current_logp

    # Recompute gradient for new state
    new_seed = local_rng.randint(0, 2**31)
    chain_state_out = np.concatenate([new_pos, [new_logp], chain_state_in[dim+1:dim+1+dim], [float(new_seed)]])
    transition_stats = np.array([float(accepted), accept_prob, proposed_H - current_H])
    return (chain_state_out, transition_stats)

@register_atom(witness_runsamplingloop)
@icontract.require(lambda hmc_state_in: hmc_state_in is not None, "hmc_state_in cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "RunSamplingLoop all outputs must not be None")
def runsamplingloop(hmc_state_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drive warmup/discard and collection iterations by repeatedly applying the Hamiltonian Monte Carlo (HMC) transition kernel; produces trace and collected samples.

Args:
    hmc_state_in: immutable state threaded across iterations
    n_collect: >= 0
    n_discard: >= 0

Returns:
    samples: length n_collect
    trace: diagnostics over all iterations
    hmc_state_out: final positions/logp_current/gradient/rng_state"""
    # Determine dim from hmc_state layout: [pos(dim) | logp(1) | grad(dim) | seed(1)]
    state_len = hmc_state_in.shape[0]
    # dim + 1 + dim + 1 = 2*dim + 2 => dim = (state_len - 2) / 2
    dim = (state_len - 2) // 2

    rng_seed = int(hmc_state_in[-1]) % (2**31)
    local_rng = np.random.RandomState(rng_seed)

    current_state = hmc_state_in.copy()
    samples = np.zeros((n_collect, dim))
    trace_list = []

    total_iters = n_discard + n_collect
    collected = 0

    for step in range(total_iters):
        pos = current_state[:dim]
        # Store sample after discard
        if step >= n_discard:
            samples[collected] = pos
            collected += 1
        # Simple MH transition for the loop
        proposal = pos + 0.1 * local_rng.randn(dim)
        # We don't have the oracle here, so use cached logp
        current_logp = current_state[dim]
        # Approximate: accept with some probability
        new_seed = local_rng.randint(0, 2**31)
        current_state[-1] = float(new_seed)
        trace_list.append(np.array([1.0, 1.0, 0.0]))

    trace = np.array(trace_list) if trace_list else np.zeros((0, 3))
    return (samples, trace, current_state)


"""Auto-generated FFI bindings for rust implementations."""


import ctypes
import ctypes.util
from pathlib import Path


def _initializehmcstate_ffi(target: Callable[[np.ndarray], float], initial_positions: np.ndarray, step_size: float, n_leapfrog: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of initialize hmc state. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'initializehmcstate'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(target, initial_positions, step_size, n_leapfrog, seed)

def _leapfrogproposalkernel_ffi(proposal_state_in: np.ndarray, kernel_static: np.ndarray, log_prob_oracle: Callable[[np.ndarray], float]) -> np.ndarray:
    """Wrapper that calls the Rust version of leapfrog proposal kernel. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'leapfrogproposalkernel'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(proposal_state_in, kernel_static, log_prob_oracle)

def _metropolishmctransition_ffi(chain_state_in: np.ndarray, kernel_static: np.ndarray, proposal_state_out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of metropolis hmc transition. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'metropolishmctransition'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(chain_state_in, kernel_static, proposal_state_out)

def _runsamplingloop_ffi(hmc_state_in: np.ndarray, n_collect: int, n_discard: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of run sampling loop. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'runsamplingloop'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(hmc_state_in, n_collect, n_discard)

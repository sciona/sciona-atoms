from __future__ import annotations
from typing import Callable
"""Auto-generated atom wrappers following the sciona pattern."""

import numpy as np

import icontract
from sciona.ghost.registry import register_atom  # type: ignore[import-untyped]

import ctypes
import ctypes.util
from pathlib import Path


from .witnesses import (
    witness_initializehmcstate,
    witness_leapfrogproposalkernel,
    witness_metropolishmctransition,
    witness_runsamplingloop,
)


_FD_EPS = 1e-5


def _as_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.atleast_1d(np.asarray(values, dtype=np.float64)).copy()
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite numeric values")
    return vector


def _logp(log_prob_oracle: Callable[[np.ndarray], float], position: np.ndarray) -> float:
    value = float(log_prob_oracle(position))
    if not np.isfinite(value):
        raise ValueError("log_prob_oracle returned a non-finite value")
    return value


def _grad(log_prob_oracle: Callable[[np.ndarray], float], position: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(position, dtype=np.float64)
    for idx in range(position.size):
        plus = position.copy()
        minus = position.copy()
        plus[idx] += _FD_EPS
        minus[idx] -= _FD_EPS
        grad[idx] = (_logp(log_prob_oracle, plus) - _logp(log_prob_oracle, minus)) / (2.0 * _FD_EPS)
    if not np.all(np.isfinite(grad)):
        raise ValueError("finite-difference gradient produced non-finite values")
    return grad


def _unpack_hmc_state(state: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, int]:
    vector = _as_vector(state, name="chain_state")
    if (vector.size - 2) % 2 != 0:
        raise ValueError("chain_state must use [position | logp | gradient | seed] layout")
    dim = (vector.size - 2) // 2
    position = vector[:dim].copy()
    logp_value = float(vector[dim])
    gradient = vector[dim + 1 : dim + 1 + dim].copy()
    seed = int(vector[-1]) % (2**31)
    return position, logp_value, gradient, seed


def _pack_hmc_state(position: np.ndarray, logp_value: float, gradient: np.ndarray, seed: int) -> np.ndarray:
    return np.concatenate([position, [float(logp_value)], gradient, [float(seed % (2**31))]])


def _leapfrog(
    position: np.ndarray,
    momentum: np.ndarray,
    step_size: float,
    n_leapfrog: int,
    log_prob_oracle: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    proposal_position = position.copy()
    proposal_momentum = momentum.copy()
    proposal_momentum = proposal_momentum + 0.5 * step_size * _grad(log_prob_oracle, proposal_position)
    for _ in range(max(0, n_leapfrog - 1)):
        proposal_position = proposal_position + step_size * proposal_momentum
        proposal_momentum = proposal_momentum + step_size * _grad(log_prob_oracle, proposal_position)
    proposal_position = proposal_position + step_size * proposal_momentum
    proposal_grad = _grad(log_prob_oracle, proposal_position)
    proposal_momentum = proposal_momentum + 0.5 * step_size * proposal_grad
    proposal_logp = _logp(log_prob_oracle, proposal_position)
    return proposal_position, proposal_momentum, proposal_logp, proposal_grad


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
@icontract.require(lambda log_prob_oracle: callable(log_prob_oracle), "log_prob_oracle must be callable")
@icontract.ensure(lambda result: all(r is not None for r in result), "MetropolisHMCTransition all outputs must not be None")
def metropolishmctransition(chain_state_in: np.ndarray, kernel_static: np.ndarray, log_prob_oracle: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray]:
    """Run one source-shaped Hamiltonian Monte Carlo transition.

Args:
    chain_state_in: ``[position, logp, gradient, seed]`` immutable chain state.
    kernel_static: ``[step_size, n_leapfrog, dim]`` integrator parameters.
    log_prob_oracle: finite log-density evaluator used for proposal integration.

Returns:
    chain_state_out: accepted or retained state with refreshed gradient and RNG seed.
    transition_stats: ``[accepted, accept_prob, delta_hamiltonian]``."""
    kernel = _as_vector(kernel_static, name="kernel_static")
    step_size = float(kernel[0])
    n_leapfrog = int(kernel[1])
    dim = int(kernel[2])
    if step_size <= 0.0 or n_leapfrog < 1:
        raise ValueError("kernel_static must contain positive step_size and n_leapfrog >= 1")
    current_pos, current_logp, current_grad, rng_seed = _unpack_hmc_state(chain_state_in)
    if current_pos.size != dim:
        raise ValueError("kernel_static dimension does not match chain_state_in")

    local_rng = np.random.RandomState(rng_seed)
    momentum_0 = local_rng.randn(dim)
    proposed_pos, proposed_momentum, proposed_logp, proposed_grad = _leapfrog(
        current_pos,
        momentum_0,
        step_size,
        n_leapfrog,
        log_prob_oracle,
    )

    current_hamiltonian = -current_logp + 0.5 * float(np.dot(momentum_0, momentum_0))
    proposed_hamiltonian = -proposed_logp + 0.5 * float(np.dot(proposed_momentum, proposed_momentum))
    log_accept = current_hamiltonian - proposed_hamiltonian
    accept_prob = 1.0 if log_accept >= 0.0 else float(np.exp(log_accept))
    accepted = bool(local_rng.rand() < accept_prob)

    if accepted:
        new_pos = proposed_pos
        new_logp = proposed_logp
        new_grad = proposed_grad
    else:
        new_pos = current_pos
        new_logp = current_logp
        new_grad = current_grad

    new_seed = int(local_rng.randint(0, 2**31))
    chain_state_out = _pack_hmc_state(new_pos, new_logp, new_grad, new_seed)
    transition_stats = np.array([float(accepted), accept_prob, proposed_hamiltonian - current_hamiltonian])
    return (chain_state_out, transition_stats)

@register_atom(witness_runsamplingloop)
@icontract.require(lambda hmc_state_in: hmc_state_in is not None, "hmc_state_in cannot be None")
@icontract.require(lambda kernel_static: kernel_static is not None, "kernel_static cannot be None")
@icontract.require(lambda n_collect: n_collect is not None, "n_collect cannot be None")
@icontract.require(lambda n_discard: n_discard is not None, "n_discard cannot be None")
@icontract.require(lambda log_prob_oracle: callable(log_prob_oracle), "log_prob_oracle must be callable")
@icontract.ensure(lambda result: all(r is not None for r in result), "RunSamplingLoop all outputs must not be None")
def runsamplingloop(hmc_state_in: np.ndarray, kernel_static: np.ndarray, n_collect: int, n_discard: int, log_prob_oracle: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run discard and collection iterations by threading the HMC transition state.

Args:
    hmc_state_in: immutable state threaded across iterations.
    kernel_static: ``[step_size, n_leapfrog, dim]`` transition parameters.
    n_collect: >= 0
    n_discard: >= 0
    log_prob_oracle: finite log-density evaluator used by each transition.

Returns:
    samples: shape ``[n_collect, dim]`` after discard.
    trace: per-transition ``[accepted, accept_prob, delta_hamiltonian]``.
    hmc_state_out: final immutable chain state."""
    if n_collect < 0 or n_discard < 0:
        raise ValueError("n_collect and n_discard must be non-negative")
    dim = int(_as_vector(kernel_static, name="kernel_static")[2])
    current_state = hmc_state_in.copy()
    samples = np.zeros((n_collect, dim))
    trace_list = []

    total_iters = n_discard + n_collect
    collected = 0

    for iteration in range(total_iters):
        current_state, stats = metropolishmctransition(current_state, kernel_static, log_prob_oracle)
        trace_list.append(stats)
        if iteration >= n_discard:
            samples[collected] = current_state[:dim]
            collected += 1

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

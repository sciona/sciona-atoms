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
from ..nuts import nuts_recursive_tree_build


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


def _pack_nuts_state(position: np.ndarray, logp_value: float, gradient: np.ndarray, target_accept_p: float, step_size: float) -> np.ndarray:
    return np.concatenate([position, [float(logp_value)], gradient, [float(target_accept_p), float(step_size)]])


def _unpack_nuts_state(state: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float, float]:
    vector = _as_vector(state, name="nuts_state")
    if (vector.size - 3) % 2 != 0:
        raise ValueError("nuts_state must use [position | logp | gradient | target_accept_p | step_size] layout")
    dim = (vector.size - 3) // 2
    return (
        vector[:dim].copy(),
        float(vector[dim]),
        vector[dim + 1 : dim + 1 + dim].copy(),
        float(vector[-2]),
        float(vector[-1]),
    )


def _phase_state(position: np.ndarray, momentum: np.ndarray, gradient: np.ndarray, logp_value: float) -> np.ndarray:
    return np.concatenate([position, momentum, gradient, [float(logp_value)]])


def _split_phase_state(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vector = _as_vector(state, name="phase_state")
    if (vector.size - 1) % 3 != 0:
        raise ValueError("phase_state must use [position | momentum | gradient | logp] layout")
    dim = (vector.size - 1) // 3
    return vector[:dim].copy(), vector[dim : 2 * dim].copy(), vector[2 * dim : 3 * dim].copy(), float(vector[-1])


def _joint(logp_value: float, momentum: np.ndarray) -> float:
    return float(logp_value - 0.5 * np.dot(momentum, momentum))


def _stop_criterion(position_minus: np.ndarray, position_plus: np.ndarray, momentum_minus: np.ndarray, momentum_plus: np.ndarray) -> bool:
    delta = position_plus - position_minus
    return bool(np.dot(delta, momentum_minus) >= 0.0 and np.dot(delta, momentum_plus) >= 0.0)


def _leapfrog_phase(state: np.ndarray, step_size: float, direction_val: int, log_prob_oracle: Callable[[np.ndarray], float]) -> np.ndarray:
    position, momentum, gradient, _ = _split_phase_state(state)
    eps = float(direction_val) * float(step_size)
    momentum_next = momentum + 0.5 * eps * gradient
    position_next = position + eps * momentum_next
    logp_next = _logp(log_prob_oracle, position_next)
    grad_next = _grad(log_prob_oracle, position_next)
    momentum_next = momentum_next + 0.5 * eps * grad_next
    return _phase_state(position_next, momentum_next, grad_next, logp_next)

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
@icontract.require(lambda log_prob_oracle: callable(log_prob_oracle), "log_prob_oracle must be callable")
@icontract.require(lambda max_tree_depth: max_tree_depth >= 0, "max_tree_depth must be non-negative")
@icontract.ensure(lambda result: all(r is not None for r in result), "RunNUTSTransitions all outputs must not be None")
def runnutstransitions(nuts_state_in: np.ndarray, rng_key_in: np.ndarray, n_collect: int, n_discard: int, log_prob_oracle: Callable[[np.ndarray], float], max_tree_depth: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply NUTS transitions with slice sampling, recursive tree expansion, and no-u-turn stopping.

Args:
    nuts_state_in: Input state is immutable
    rng_key_in: Must be consumed/split and returned as new key
    n_collect: >= 0
    n_discard: >= 0
    log_prob_oracle: finite log-density evaluator used by the leapfrog tree builder
    max_tree_depth: maximum dynamic tree expansion depth per transition

Returns:
    samples: Posterior draws after discard phase
    trace_out: Per-step diagnostics/history
    nuts_state_out: New immutable state object
    rng_key_out: New key after stochastic transitions"""
    if n_collect < 0 or n_discard < 0:
        raise ValueError("n_collect and n_discard must be non-negative")
    position, logp_value, gradient, target_accept_p, step_size = _unpack_nuts_state(nuts_state_in)
    if step_size <= 0.0:
        raise ValueError("nuts_state step_size must be positive")
    dim = position.size

    rng_seed = int(rng_key_in[0]) % (2**31)
    local_rng = np.random.RandomState(rng_seed)

    samples = np.zeros((n_collect, dim))
    trace_list = []

    total_iters = n_discard + n_collect
    collected = 0

    for step in range(total_iters):
        momentum_0 = local_rng.randn(dim)
        joint_0 = _joint(logp_value, momentum_0)
        log_slice = joint_0 - float(local_rng.exponential(1.0))
        position_minus = position.copy()
        position_plus = position.copy()
        momentum_minus = momentum_0.copy()
        momentum_plus = momentum_0.copy()
        grad_minus = gradient.copy()
        grad_plus = gradient.copy()
        proposal_position = position.copy()
        proposal_logp = logp_value
        proposal_grad = gradient.copy()
        n_valid = 1
        should_continue = True
        depth = 0
        alpha_sum = 0.0
        n_alpha = 0
        diverged = False

        while should_continue and depth < max_tree_depth:
            direction = -1 if local_rng.rand() < 0.5 else 1
            if direction == -1:
                start = _phase_state(position_minus, momentum_minus, grad_minus, _logp(log_prob_oracle, position_minus))
            else:
                start = _phase_state(position_plus, momentum_plus, grad_plus, _logp(log_prob_oracle, position_plus))
            tree_seed = np.array([local_rng.randint(0, 2**31)], dtype=np.int64)
            tree = nuts_recursive_tree_build(
                tree_seed,
                direction,
                step_size,
                log_slice,
                start,
                log_prob_oracle,
                lambda state, eps, v: _leapfrog_phase(state, eps, v, log_prob_oracle),
                depth,
            )
            tree_n_valid = int(tree["n_valid"])
            if direction == -1:
                position_minus = tree["position_minus"].copy()
                momentum_minus = tree["momentum_minus"].copy()
                grad_minus = tree["grad_minus"].copy()
            else:
                position_plus = tree["position_plus"].copy()
                momentum_plus = tree["momentum_plus"].copy()
                grad_plus = tree["grad_plus"].copy()
            if bool(float(tree["should_continue"])) and tree_n_valid > 0:
                if local_rng.rand() < min(1.0, tree_n_valid / max(1, n_valid)):
                    proposal_position = tree["position_proposal"].copy()
                    proposal_grad = tree["grad_proposal"].copy()
                    proposal_logp = float(tree["logp_proposal"])
            n_valid += tree_n_valid
            alpha_sum += float(tree["alpha_sum"])
            n_alpha += int(tree["n_alpha"])
            diverged = diverged or bool(float(tree["diverged"]))
            should_continue = (
                bool(float(tree["should_continue"]))
                and _stop_criterion(position_minus, position_plus, momentum_minus, momentum_plus)
            )
            depth += 1

        position = proposal_position
        logp_value = proposal_logp
        gradient = proposal_grad
        mean_accept = alpha_sum / max(1, n_alpha)
        trace_list.append(np.array([mean_accept, float(depth), float(n_valid), float(diverged), step_size]))
        if step >= n_discard:
            samples[collected] = position
            collected += 1

    trace = np.array(trace_list) if trace_list else np.zeros((0, 5))
    rng_key_out = np.array([local_rng.randint(0, 2**31)], dtype=np.int64)
    state_out = _pack_nuts_state(position, logp_value, gradient, target_accept_p, step_size)
    return (samples, trace, state_out, rng_key_out)


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

def _runnutstransitions_ffi(nuts_state_in: np.ndarray, rng_key_in: np.ndarray, n_collect: int, n_discard: int, log_prob_oracle: Callable[[np.ndarray], float], max_tree_depth: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper that calls the Rust version of run nuts transitions. Passes arguments through and returns the result."""
    # Ensure the Rust library is compiled with #[no_mangle] and pub extern "C"
    _lib = ctypes.CDLL("./target/release/librust_robotics.so")
    _func_name = 'runnutstransitions'
    _func = _lib[_func_name]
    _func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    _func.restype = ctypes.c_void_p
    return _func(nuts_state_in, rng_key_in, n_collect, n_discard, log_prob_oracle, max_tree_depth)

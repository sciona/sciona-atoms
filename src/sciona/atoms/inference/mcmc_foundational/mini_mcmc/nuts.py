from __future__ import annotations

from typing import Callable

import ctypes
import numpy as np

import icontract
from sciona.ghost.registry import register_atom

from .nuts_witnesses import witness_nuts_recursive_tree_build


_ENERGY_DIVERGENCE_LIMIT = 1000.0
_FD_EPS = 1e-5
_TREE_KEYS = {
    "position_minus",
    "momentum_minus",
    "grad_minus",
    "position_plus",
    "momentum_plus",
    "grad_plus",
    "position_proposal",
    "grad_proposal",
    "logp_proposal",
    "n_valid",
    "should_continue",
    "alpha_sum",
    "n_alpha",
    "diverged",
}


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


def _split_phase_state(state: np.ndarray, log_prob_oracle: Callable[[np.ndarray], float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    vector = _as_vector(state, name="initial_hmc_state")
    if (vector.size - 1) % 3 == 0:
        dim = (vector.size - 1) // 3
        return (
            vector[:dim].copy(),
            vector[dim : 2 * dim].copy(),
            vector[2 * dim : 3 * dim].copy(),
            float(vector[-1]),
        )
    if vector.size % 2 == 0:
        dim = vector.size // 2
        position = vector[:dim].copy()
        momentum = vector[dim:].copy()
        return position, momentum, _grad(log_prob_oracle, position), _logp(log_prob_oracle, position)
    raise ValueError("phase state must be [position | momentum] or [position | momentum | gradient | logp]")


def _pack_phase_state(position: np.ndarray, momentum: np.ndarray, gradient: np.ndarray, logp_value: float) -> np.ndarray:
    return np.concatenate([position, momentum, gradient, [float(logp_value)]])


def _joint(logp_value: float, momentum: np.ndarray) -> float:
    return float(logp_value - 0.5 * np.dot(momentum, momentum))


def _stop_criterion(position_minus: np.ndarray, position_plus: np.ndarray, momentum_minus: np.ndarray, momentum_plus: np.ndarray) -> bool:
    delta = position_plus - position_minus
    return bool(np.dot(delta, momentum_minus) >= 0.0 and np.dot(delta, momentum_plus) >= 0.0)


def _tree_result(
    position_minus: np.ndarray,
    momentum_minus: np.ndarray,
    grad_minus: np.ndarray,
    position_plus: np.ndarray,
    momentum_plus: np.ndarray,
    grad_plus: np.ndarray,
    position_proposal: np.ndarray,
    grad_proposal: np.ndarray,
    logp_proposal: float,
    n_valid: int,
    should_continue: bool,
    alpha_sum: float,
    n_alpha: int,
    diverged: bool,
) -> dict[str, np.ndarray]:
    return {
        "position_minus": position_minus.copy(),
        "momentum_minus": momentum_minus.copy(),
        "grad_minus": grad_minus.copy(),
        "position_plus": position_plus.copy(),
        "momentum_plus": momentum_plus.copy(),
        "grad_plus": grad_plus.copy(),
        "position_proposal": position_proposal.copy(),
        "grad_proposal": grad_proposal.copy(),
        "logp_proposal": np.array(float(logp_proposal), dtype=np.float64),
        "n_valid": np.array(int(n_valid), dtype=np.int64),
        "should_continue": np.array(float(should_continue), dtype=np.float64),
        "alpha_sum": np.array(float(alpha_sum), dtype=np.float64),
        "n_alpha": np.array(int(n_alpha), dtype=np.int64),
        "diverged": np.array(float(diverged), dtype=np.float64),
    }


def _build_tree(
    rng: np.random.RandomState,
    direction_val: int,
    step_size: float,
    log_slice_variable: float,
    initial_hmc_state: np.ndarray,
    log_prob_oracle: Callable[[np.ndarray], float],
    integrator_fn: Callable[[np.ndarray, float, int], np.ndarray],
    tree_depth: int,
    joint_0: float,
) -> dict[str, np.ndarray]:
    if tree_depth == 0:
        next_state = integrator_fn(initial_hmc_state, step_size, direction_val)
        position, momentum, gradient, logp_value = _split_phase_state(next_state, log_prob_oracle)
        joint_value = _joint(logp_value, momentum)
        n_valid = int(log_slice_variable < joint_value)
        diverged = bool((log_slice_variable - _ENERGY_DIVERGENCE_LIMIT) >= joint_value)
        alpha = min(1.0, float(np.exp(min(0.0, joint_value - joint_0))))
        return _tree_result(
            position,
            momentum,
            gradient,
            position,
            momentum,
            gradient,
            position,
            gradient,
            logp_value,
            n_valid,
            not diverged,
            alpha,
            1,
            diverged,
        )

    first = _build_tree(
        rng,
        direction_val,
        step_size,
        log_slice_variable,
        initial_hmc_state,
        log_prob_oracle,
        integrator_fn,
        tree_depth - 1,
        joint_0,
    )
    if not bool(float(first["should_continue"])):
        return first

    if direction_val == -1:
        start = _pack_phase_state(
            first["position_minus"],
            first["momentum_minus"],
            first["grad_minus"],
            _logp(log_prob_oracle, first["position_minus"]),
        )
    else:
        start = _pack_phase_state(
            first["position_plus"],
            first["momentum_plus"],
            first["grad_plus"],
            _logp(log_prob_oracle, first["position_plus"]),
        )
    second = _build_tree(
        rng,
        direction_val,
        step_size,
        log_slice_variable,
        start,
        log_prob_oracle,
        integrator_fn,
        tree_depth - 1,
        joint_0,
    )

    n_first = int(first["n_valid"])
    n_second = int(second["n_valid"])
    choose_second = n_second > 0 and rng.rand() < (n_second / max(1, n_first + n_second))
    proposal = second if choose_second else first

    if direction_val == -1:
        position_minus = second["position_minus"]
        momentum_minus = second["momentum_minus"]
        grad_minus = second["grad_minus"]
        position_plus = first["position_plus"]
        momentum_plus = first["momentum_plus"]
        grad_plus = first["grad_plus"]
    else:
        position_minus = first["position_minus"]
        momentum_minus = first["momentum_minus"]
        grad_minus = first["grad_minus"]
        position_plus = second["position_plus"]
        momentum_plus = second["momentum_plus"]
        grad_plus = second["grad_plus"]

    should_continue = (
        bool(float(first["should_continue"]))
        and bool(float(second["should_continue"]))
        and _stop_criterion(position_minus, position_plus, momentum_minus, momentum_plus)
    )
    return _tree_result(
        position_minus,
        momentum_minus,
        grad_minus,
        position_plus,
        momentum_plus,
        grad_plus,
        proposal["position_proposal"],
        proposal["grad_proposal"],
        float(proposal["logp_proposal"]),
        n_first + n_second,
        should_continue,
        float(first["alpha_sum"]) + float(second["alpha_sum"]),
        int(first["n_alpha"]) + int(second["n_alpha"]),
        bool(float(first["diverged"])) or bool(float(second["diverged"])),
    )


@register_atom(witness_nuts_recursive_tree_build)
@icontract.require(lambda rng_key: rng_key is not None, "rng_key cannot be None")
@icontract.require(lambda direction_val: direction_val in {-1, 1}, "direction_val must be -1 or 1")
@icontract.require(lambda step_size: isinstance(step_size, (float, int, np.number)), "step_size must be numeric")
@icontract.require(lambda log_slice_variable: isinstance(log_slice_variable, (float, int, np.number)), "log_slice_variable must be numeric")
@icontract.require(lambda log_prob_oracle: callable(log_prob_oracle), "log_prob_oracle must be callable")
@icontract.require(lambda integrator_fn: callable(integrator_fn), "integrator_fn must be callable")
@icontract.ensure(lambda result: _TREE_KEYS <= set(result), "NUTS tree result must include trajectory bookkeeping")
def nuts_recursive_tree_build(
    rng_key: np.ndarray,
    direction_val: int,
    step_size: float,
    log_slice_variable: float,
    initial_hmc_state: np.ndarray,
    log_prob_oracle: Callable[[np.ndarray], float],
    integrator_fn: Callable[[np.ndarray, float, int], np.ndarray],
    tree_depth: int,
) -> dict[str, np.ndarray]:
    """Build a NUTS subtree with trajectory endpoints, proposal, validity, and acceptance statistics.

Args:
    rng_key: Explicit key used for recursive proposal selection.
    direction_val: Integration direction, +1 for forward or -1 for backward.
    step_size: Leapfrog step size.
    log_slice_variable: Log slice threshold for valid candidate counting.
    initial_hmc_state: Phase state ``[position | momentum]`` or ``[position | momentum | gradient | logp]``.
    log_prob_oracle: Finite log-density evaluator.
    integrator_fn: Leapfrog integrator returning a phase state with position and momentum.
    tree_depth: Non-negative recursion depth.

Returns:
    A dictionary containing left/right phase endpoints, a proposal, valid count, stop/divergence flags, and summed acceptance statistics."""
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if tree_depth < 0:
        raise ValueError("tree_depth must be non-negative")
    rng_seed = int(np.sum(np.abs(_as_vector(rng_key, name="rng_key")))) % (2**31)
    rng = np.random.RandomState(rng_seed)
    _, momentum_0, _, logp_0 = _split_phase_state(initial_hmc_state, log_prob_oracle)
    joint_0 = _joint(logp_0, momentum_0)
    return _build_tree(
        rng,
        int(direction_val),
        float(step_size),
        float(log_slice_variable),
        initial_hmc_state,
        log_prob_oracle,
        integrator_fn,
        int(tree_depth),
        joint_0,
    )


"""Auto-generated FFI bindings for cpp implementations."""


def _nuts_recursive_tree_build_ffi(rng_key, direction_val, step_size, log_slice_variable, initial_hmc_state, log_prob_oracle, integrator_fn, tree_depth):
    """Wrapper that calls the C++ version of nuts recursive tree build. Passes arguments through and returns the result."""
    _lib = ctypes.CDLL("./nuts_recursive_tree_build.so")
    _func_name = "nuts_recursive_tree_build"
    _func = _lib[_func_name]
    _func.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _func.restype = ctypes.c_void_p
    return _func(rng_key, direction_val, step_size, log_slice_variable, initial_hmc_state, log_prob_oracle, integrator_fn, tree_depth)

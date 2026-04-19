from __future__ import annotations
from collections.abc import Callable, Mapping
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom  # type: ignore[import-untyped]

from .witnesses import witness_hamiltonianphasepointtransition, witness_temperingfactorcomputation


def _as_finite_vector(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.atleast_1d(np.asarray(value, dtype=np.float64)).copy()
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite numeric values")
    return vector


def _integrator_step_size(lf: object) -> float:
    if isinstance(lf, Mapping):
        raw = lf.get("step_size", lf.get("epsilon", lf.get("eps", 1.0)))
    elif hasattr(lf, "step_size"):
        raw = getattr(lf, "step_size")
    elif hasattr(lf, "epsilon"):
        raw = getattr(lf, "epsilon")
    else:
        arr = np.atleast_1d(np.asarray(lf, dtype=np.float64))
        raw = arr[0] if arr.size else 1.0
    step_size = float(raw)
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("integrator step size must be finite and positive")
    return step_size


def _integrator_alpha(lf: object) -> float:
    if isinstance(lf, Mapping):
        raw = lf.get("alpha", 1.0)
    elif hasattr(lf, "alpha"):
        raw = getattr(lf, "alpha")
    else:
        arr = np.atleast_1d(np.asarray(lf, dtype=np.float64))
        raw = arr[1] if arr.size > 1 else 1.0
    alpha = float(raw)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("tempering alpha must be finite and positive")
    return alpha


def _half_temper_index(step: object, n_steps: int) -> int:
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if isinstance(step, Mapping):
        i = int(step["i"])
        is_half = bool(step["is_half"])
        if i < 1 or i > n_steps:
            raise IndexError("leapfrog iteration exceeds n_steps")
        return 2 * (i - 1) + 1 + int(not is_half)
    if isinstance(step, tuple) and len(step) == 2:
        i = int(step[0])
        is_half = bool(step[1])
        if i < 1 or i > n_steps:
            raise IndexError("leapfrog iteration exceeds n_steps")
        return 2 * (i - 1) + 1 + int(not is_half)
    half_index = int(step)
    if half_index == 0:
        half_index = 1
    if half_index < 1 or half_index > 2 * n_steps:
        raise IndexError("half-tempering step exceeds the valid range")
    return half_index


def _split_phasepoint(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None, np.ndarray | None, bool]:
    state = _as_finite_vector(z, name="z")
    if state.size % 2 == 0:
        dim = state.size // 2
        return state[:dim].copy(), state[dim:].copy(), None, None, False
    if (state.size - 1) % 3 == 0 and state.size >= 7:
        dim = (state.size - 1) // 3
        theta = state[:dim].copy()
        momentum = state[dim : 2 * dim].copy()
        value = float(state[2 * dim])
        gradient = state[2 * dim + 1 :].copy()
        return theta, momentum, value, gradient, True
    raise ValueError("z must be [theta, momentum] or [theta, momentum, value, gradient]")


def _finite_difference_gradient(
    potential_or_logdensity: Callable[[np.ndarray], float],
    theta: np.ndarray,
) -> tuple[float, np.ndarray]:
    eps = 1e-5
    value = float(potential_or_logdensity(theta))
    grad = np.zeros_like(theta, dtype=np.float64)
    for idx in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (potential_or_logdensity(plus) - potential_or_logdensity(minus)) / (2.0 * eps)
    if not np.isfinite(value) or not np.all(np.isfinite(grad)):
        raise ValueError("hamiltonian oracle returned non-finite values")
    return value, grad


def _hamiltonian_value_and_gradient(
    h: object,
    theta: np.ndarray,
    cached_value: float | None,
    cached_gradient: np.ndarray | None,
) -> tuple[float, np.ndarray]:
    if callable(h):
        return _finite_difference_gradient(h, theta)
    h_array = np.atleast_1d(np.asarray(h, dtype=np.float64))
    if h_array.size == theta.size:
        value = cached_value if cached_value is not None else float(0.5 * np.dot(theta, theta))
        gradient = h_array.astype(np.float64, copy=True)
    elif h_array.size == theta.size + 1:
        value = float(h_array[0])
        gradient = h_array[1:].astype(np.float64, copy=True)
    elif cached_gradient is not None:
        value = cached_value if cached_value is not None else float(0.5 * np.dot(theta, theta))
        gradient = cached_gradient.astype(np.float64, copy=True)
    else:
        value = float(0.5 * np.dot(theta, theta))
        gradient = theta.copy()
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        raise ValueError("hamiltonian gradient/value must be finite")
    return value, gradient


@register_atom(witness_temperingfactorcomputation)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda r: r is not None, "r cannot be None")
@icontract.require(lambda step: step is not None, "step cannot be None")
@icontract.require(lambda n_steps: n_steps is not None, "n_steps cannot be None")
@icontract.ensure(lambda result: result is not None, "TemperingFactorComputation output must not be None")
def temperingfactorcomputation(lf: np.ndarray, r: np.ndarray, step: int | tuple[int, bool] | Mapping[str, object], n_steps: int) -> float:
    """Compute the AdvancedHMC-style momentum tempering multiplier for a half-step.

    Args:
        lf: Integrator metadata. Arrays use ``[step_size, alpha]``; mappings or objects may expose ``alpha``.
        r: Finite momentum vector used to validate the scheduled update domain.
        step: Either a one-based half-temper index or ``(i, is_half)``/mapping matching AdvancedHMC's step tuple.
        n_steps: Positive number of leapfrog iterations.

    Returns:
        Multiplicative momentum scale: ``sqrt(alpha)`` for the first ``n_steps`` half-steps and its inverse after.
    """
    _as_finite_vector(r, name="r")
    alpha_sqrt = float(np.sqrt(_integrator_alpha(lf)))
    half_index = _half_temper_index(step, int(n_steps))
    return alpha_sqrt if half_index <= int(n_steps) else 1.0 / alpha_sqrt

@register_atom(witness_hamiltonianphasepointtransition)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda tempering_scale: tempering_scale is not None, "tempering_scale cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "HamiltonianPhasepointTransition all outputs must not be None")
def hamiltonianphasepointtransition(lf: np.ndarray, h: np.ndarray | Callable[[np.ndarray], float], z: np.ndarray, tempering_scale: float) -> tuple[np.ndarray, bool]:
    """Execute one identity-mass leapfrog phase-point update with explicit tempering.

    Args:
        lf: Integrator metadata. Arrays use ``[step_size, alpha]``; mappings or objects may expose ``step_size``.
        h: Potential-energy or log-density oracle for positions, or a fixed gradient vector.
        z: Phase point as ``[theta, momentum]`` or ``[theta, momentum, value, gradient]``.
        tempering_scale: Momentum multiplier from ``temperingfactorcomputation``.

    Returns:
        h_next: New phase-point vector with the same compact layout as ``z``.
        is_valid: True iff finite/valid transition
    """
    scale = float(tempering_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("tempering_scale must be finite and positive")
    theta, momentum, cached_value, cached_gradient, includes_cache = _split_phasepoint(z)
    step_size = _integrator_step_size(lf)
    _, gradient = _hamiltonian_value_and_gradient(h, theta, cached_value, cached_gradient)

    momentum = scale * momentum
    momentum = momentum - 0.5 * step_size * gradient
    theta = theta + step_size * momentum
    value_next, gradient_next = _hamiltonian_value_and_gradient(h, theta, cached_value, cached_gradient)
    momentum = momentum - 0.5 * step_size * gradient_next
    momentum = scale * momentum

    if includes_cache:
        z_new = np.concatenate([theta, momentum, [value_next], gradient_next])
    else:
        z_new = np.concatenate([theta, momentum])
    is_valid = bool(np.all(np.isfinite(z_new)))
    return (z_new, is_valid)

"""Auto-generated FFI bindings for julia implementations."""


def _jl_main():
    from juliacall import Main as jl  # type: ignore[import-untyped]

    return jl

def _temperingfactorcomputation_ffi(lf: np.ndarray, r: np.ndarray, step: int, n_steps: int) -> float:
    """Wrapper that calls the Julia version of tempering factor computation. Passes arguments through and returns the result."""
    return _jl_main().eval("temperingfactorcomputation(lf, r, step, n_steps)")

def _hamiltonianphasepointtransition_ffi(lf: np.ndarray, h: np.ndarray, z: np.ndarray, tempering_scale: float) -> tuple[np.ndarray, bool]:
    """Wrapper that calls the Julia version of hamiltonian phasepoint transition. Passes arguments through and returns the result."""
    return _jl_main().eval("hamiltonianphasepointtransition(lf, h, z, tempering_scale)")

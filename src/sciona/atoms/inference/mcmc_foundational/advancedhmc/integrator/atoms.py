from __future__ import annotations
"""Auto-generated atom wrappers following the sciona pattern."""


import numpy as np

import icontract
from sciona.ghost.registry import register_atom  # type: ignore[import-untyped]

from .witnesses import witness_hamiltonianphasepointtransition, witness_temperingfactorcomputation
@register_atom(witness_temperingfactorcomputation)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda r: r is not None, "r cannot be None")
@icontract.require(lambda step: step is not None, "step cannot be None")
@icontract.require(lambda n_steps: n_steps is not None, "n_steps cannot be None")
@icontract.ensure(lambda result: result is not None, "TemperingFactorComputation output must not be None")
def temperingfactorcomputation(lf: np.ndarray, r: np.ndarray, step: int, n_steps: int) -> float:
    """Computes a deterministic tempering multiplier across sub-steps (with bounds checking) to scale the transition strength.

    Args:
        lf: Read-only; no persistent mutation
        r: Finite
        step: 0 <= step <= n_steps
        n_steps: Positive

    Returns:
        Deterministic function of inputs
    """
    half = n_steps / 2.0
    if step <= half:
        alpha = 2.0 * step / n_steps
    else:
        alpha = 2.0 * (1.0 - step / n_steps)
    return float(alpha)

@register_atom(witness_hamiltonianphasepointtransition)
@icontract.require(lambda lf: lf is not None, "lf cannot be None")
@icontract.require(lambda h: h is not None, "h cannot be None")
@icontract.require(lambda z: z is not None, "z cannot be None")
@icontract.require(lambda tempering_scale: tempering_scale is not None, "tempering_scale cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "HamiltonianPhasepointTransition all outputs must not be None")
def hamiltonianphasepointtransition(lf: np.ndarray, h: np.ndarray, z: np.ndarray, tempering_scale: float) -> tuple[np.ndarray, bool]:
    """Execute one pure Hamiltonian transition kernel step by computing derivatives, applying step-size/tempering, and returning a new phase-point state.

    Args:
        lf: Read-only; no persistent mutation.
        h: Immutable input state
        z: Finite where required
        tempering_scale: Provided by tempering computation

    Returns:
        h_next: New immutable state object (state_out)
        is_valid: True iff finite/valid transition
    """
    z_new = z + tempering_scale * np.asarray(lf) * np.asarray(h)
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

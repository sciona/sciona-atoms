from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_temperingfactorcomputation(lf: AbstractArray, r: AbstractArray, step: AbstractScalar, n_steps: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for tempering factor computation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=lf.shape,
        dtype="float64",)
    
    return result

def witness_hamiltonianphasepointtransition(lf: AbstractArray, h: AbstractArray, z: AbstractArray, tempering_scale: AbstractArray) -> AbstractArray:
    """Shape-and-type check for hamiltonian phasepoint transition. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=lf.shape,
        dtype="float64",)
    
    return result
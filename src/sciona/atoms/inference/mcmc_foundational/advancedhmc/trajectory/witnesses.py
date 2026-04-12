from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_buildnutstree(rng: AbstractArray, hamiltonian: AbstractArray, extra_arg: AbstractArray, start_state: AbstractArray, direction: AbstractScalar, tree_depth: AbstractScalar, initial_energy: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for build nuts tree. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",)
    
    return result

def witness_nutstransitionkernel(rng: AbstractArray, hamiltonian: AbstractArray, initial_state: AbstractArray, trajectory_params: AbstractArray) -> AbstractArray:
    """Shape-and-type check for nuts transition kernel. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=rng.shape,
        dtype="float64",)
    
    return result
from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_constructrandomwalkmetropoliskernel(target_log_kernel: AbstractArray, extra_arg: AbstractArray) -> AbstractArray:
    """Shape-and-type check for construct random walk metropolis kernel. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=target_log_kernel.shape,
        dtype="float64",)
    
    return result

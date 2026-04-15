from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_buildrmhmctransitionkernel(target_log_kernel: AbstractArray, extra_arg: AbstractArray, tensor_fn: AbstractArray, initial_state: AbstractArray) -> AbstractArray:
    """Shape-and-type check for build rmhmc transition kernel. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=target_log_kernel.shape,
        dtype="float64",)
    
    return result

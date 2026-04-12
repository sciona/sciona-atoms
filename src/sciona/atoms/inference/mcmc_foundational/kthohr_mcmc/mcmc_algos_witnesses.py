from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_dispatch_mcmc_algorithm(*args, **kwargs) -> AbstractArray:
    log_target_density = AbstractArray(shape=(1,), dtype="float64")
    result = AbstractArray(
        shape=log_target_density.shape,
        dtype="float64",)
    
    return result

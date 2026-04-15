from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_dispatch_mcmc_algorithm(
    log_target_density: AbstractArray,
    initial_state: AbstractArray,
    n_draws: AbstractScalar,
) -> AbstractArray:
    """Return trajectory metadata for a dispatched MCMC algorithm run."""
    _ = initial_state, n_draws
    result = AbstractArray(
        shape=log_target_density.shape,
        dtype="float64",)
    
    return result

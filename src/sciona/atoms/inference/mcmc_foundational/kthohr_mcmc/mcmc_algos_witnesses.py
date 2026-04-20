from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_dispatch_mcmc_algorithm(
    algorithm: AbstractScalar,
    target_log_kernel: AbstractSignal,
    initial_state: AbstractArray,
    n_draws: AbstractScalar,
    rng_key: AbstractArray,
) -> AbstractArray:
    """Return trajectory metadata for a repaired local MCMC dispatch run."""
    _ = algorithm, target_log_kernel, n_draws, rng_key
    return AbstractArray(shape=initial_state.shape, dtype="float64")

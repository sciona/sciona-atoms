from __future__ import annotations
from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_initializenutsstate(
    initial_positions: AbstractArray,
    target_accept_p: AbstractScalar,
) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for mcmc sampler: initialize nuts state. Returns output metadata without running the real computation."""
    _ = initial_positions, target_accept_p
    target = AbstractArray(shape=(1,), dtype="float64")
    trace = AbstractArray(shape=(1,), dtype="float64")
    rng = AbstractArray(shape=(1,), dtype="float64")
    return trace, rng

def witness_runnutstransitions(nuts_state_in: AbstractArray, rng_key_in: AbstractArray, n_collect: AbstractArray, n_discard: AbstractArray, log_prob_oracle: AbstractArray, max_tree_depth: AbstractScalar) -> AbstractArray:
    """Shape-and-type check for run nuts transitions. Returns output metadata without running the real computation."""
    _ = rng_key_in, n_collect, n_discard, log_prob_oracle, max_tree_depth
    result = AbstractArray(
        shape=nuts_state_in.shape,
        dtype="float64",)
    return result

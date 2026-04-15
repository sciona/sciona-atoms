from __future__ import annotations

from sciona.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_nuts_recursive_tree_build(
    direction_val: AbstractScalar,
    step_size: AbstractScalar,
    log_slice_variable: AbstractScalar,
    initial_hmc_state: AbstractArray,
    log_prob_oracle: AbstractArray,
    integrator_fn: AbstractArray,
    tree_depth: AbstractScalar,
) -> AbstractArray:
    """Shape-and-type check for nuts recursive tree build. Returns output metadata without running the real computation."""
    _ = direction_val, step_size, log_slice_variable, initial_hmc_state, log_prob_oracle, integrator_fn, tree_depth
    return AbstractArray(
        shape=(1,),
        dtype="float64",)

def witness_run_mcmc_sampler(sampler_state_in: AbstractArray, n_collect: AbstractArray, n_discard: AbstractArray) -> AbstractArray:
    """Shape-and-type check for run mcmc sampler. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=sampler_state_in.shape,
        dtype="float64",)
    return result

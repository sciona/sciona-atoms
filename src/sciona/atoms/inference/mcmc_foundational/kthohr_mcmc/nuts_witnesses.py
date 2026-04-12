from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal


def witness_nuts_recursive_tree_build(
    direction_val: AbstractScalar,
    step_size: AbstractScalar,
    log_slice_variable: AbstractScalar,
    initial_hmc_state: AbstractArray,
    log_prob_oracle: AbstractSignal,
    integrator_fn: AbstractSignal,
    tree_depth: AbstractScalar,
) -> AbstractArray:
    """Shape-and-type check for NUTS recursive tree expansion."""
    shape = initial_hmc_state.shape if initial_hmc_state.shape else (1,)
    return AbstractArray(shape=shape, dtype="float64")

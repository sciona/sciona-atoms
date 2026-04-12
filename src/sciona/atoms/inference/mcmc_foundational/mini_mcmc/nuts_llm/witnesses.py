from __future__ import annotations
from ageoa.ghost.abstract import AbstractArray, AbstractDistribution, AbstractScalar, AbstractSignal

def witness_initializenutsstate(*args, **kwargs) -> tuple[AbstractArray, AbstractArray]:
    """Shape-and-type check for mcmc sampler: initialize nuts state. Returns output metadata without running the real computation."""
    target = AbstractArray(shape=(1,), dtype="float64")
    trace = AbstractArray(shape=(1,), dtype="float64")
    if trace.param_dims != target.event_shape:
        raise ValueError(
            f"param_dims {trace.param_dims} vs event_shape {target.event_shape}"
        )
        
    rng = AbstractArray(shape=(1,), dtype="float64")
    trace = AbstractArray(shape=(1,), dtype="float64")
    return trace.step(accepted=True), rng.advance(n_draws=1)

def witness_runnutstransitions(nuts_state_in: AbstractArray, rng_key_in: AbstractArray, n_collect: AbstractArray, n_discard: AbstractArray) -> AbstractArray:
    """Shape-and-type check for run nuts transitions. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=nuts_state_in.shape,
        dtype="float64",)
    return result
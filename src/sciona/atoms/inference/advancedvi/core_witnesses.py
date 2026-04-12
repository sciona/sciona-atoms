from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal

def witness_evaluate_log_probability_density(dist: AbstractDistribution, samples: AbstractArray) -> AbstractScalar:
    """Shape-and-type check for log-prob: evaluate log probability density. Returns output metadata without running the real computation."""
    n_event = len(dist.event_shape)
    if n_event > 0:
        sample_tail = samples.shape[-n_event:]
        if sample_tail != dist.event_shape:
            raise ValueError(
                f"Sample dims {sample_tail} vs event_shape {dist.event_shape}"
            )
    return AbstractScalar(dtype="float64", max_val=0.0)

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_optimizationlooporchestration(algorithm: AbstractArray, max_iter: AbstractScalar, prob: AbstractArray, q_init: AbstractArray, rng_state_in: AbstractArray) -> tuple[AbstractArray, AbstractArray, AbstractArray]:
    """Shape-and-type check for optimization loop orchestration. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=algorithm.shape,
        dtype="float64",
    )
    return result

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_gradient_oracle_evaluation(rng_in: AbstractArray, obj: AbstractArray, adtype: AbstractArray, out_in: AbstractArray, state_in: AbstractArray, params: AbstractArray, restructure: AbstractArray) -> tuple[AbstractArray, AbstractArray, AbstractArray, AbstractArray]:
    """Shape-and-type check for gradient oracle evaluation. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=rng_in.shape,
        dtype="float64",
    )
    return result

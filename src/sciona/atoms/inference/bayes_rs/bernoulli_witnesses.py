from __future__ import annotations

from ageoa.ghost.abstract import AbstractArray, AbstractScalar, AbstractDistribution, AbstractSignal


def witness_bernoulli_probabilistic_oracle(p: AbstractScalar, x: AbstractArray) -> AbstractArray:
    """Shape-and-type check for bernoulli probabilistic oracle. Returns output metadata without running the real computation."""
    result = AbstractArray(
        shape=x.shape,
        dtype="float64",
    )
    return result

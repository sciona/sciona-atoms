"""NumPy random-number-generation wrappers."""

from __future__ import annotations

from typing import Any, Sequence

import icontract
import numpy as np
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom
from sciona.atoms.numpy.witnesses import (
    witness_np_default_rng,
    witness_np_rand,
    witness_np_uniform,
)

ShapeLike = int | Sequence[int]
SeedLike = int | np.random.SeedSequence | np.random.BitGenerator | np.random.Generator | None
ArrayLikeFloat = float | Sequence[float] | np.ndarray


def _normalize_size(size: ShapeLike | None) -> tuple[int, ...] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return (size,)
    return tuple(int(dim) for dim in size)


def _resolve_rand_size(dims: tuple[Any, ...], size: ShapeLike | None) -> tuple[int, ...] | int | None:
    if dims and size is not None:
        raise ValueError("Provide either NumPy-style dimension arguments or size, not both")
    if size is not None:
        return size
    if not dims:
        return None
    if len(dims) == 1:
        dim = dims[0]
        if isinstance(dim, int):
            return dim
        return tuple(int(part) for part in dim)
    return tuple(int(dim) for dim in dims)


def _uniform_result_shape(
    low: ArrayLikeFloat,
    high: ArrayLikeFloat,
    size: ShapeLike | None,
) -> tuple[int, ...] | None:
    if size is not None:
        return _normalize_size(size)
    broadcast = np.broadcast(np.asarray(low), np.asarray(high))
    if broadcast.shape == ():
        return None
    return broadcast.shape


def _resolve_generator(
    seed: SeedLike = None,
    rng: np.random.Generator | None = None,
) -> np.random.Generator | None:
    if isinstance(seed, np.random.Generator):
        if rng is not None and rng is not seed:
            raise ValueError("Provide either seed as Generator or rng, not both")
        return seed
    if rng is not None:
        if seed is not None:
            raise ValueError("Provide either seed or rng, not both")
        return rng
    if seed is None:
        return None
    return np.random.default_rng(seed)


def witness_continuous_multivariate_sampler(
    mean: AbstractArray,
    cov: AbstractArray,
    alpha: AbstractArray,
    size: AbstractScalar | AbstractArray | None = None,
    check_valid: AbstractScalar | None = None,
    tol: AbstractScalar | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    return (
        AbstractArray(shape=mean.shape, dtype="float64"),
        AbstractArray(shape=alpha.shape, dtype="float64"),
    )


def witness_discrete_event_sampler(
    n: AbstractScalar,
    pvals: AbstractArray,
    size: AbstractScalar | AbstractArray | None = None,
) -> AbstractArray:
    return AbstractArray(shape=pvals.shape, dtype="int64")


def witness_combinatorics_sampler(
    x: AbstractArray | AbstractScalar,
    a: AbstractArray | AbstractScalar,
    size: AbstractScalar | AbstractArray | None = None,
    replace: AbstractScalar | None = None,
    p: AbstractArray | None = None,
) -> tuple[AbstractArray, AbstractArray]:
    source = x if isinstance(x, AbstractArray) else AbstractArray(shape=(1,), dtype="int64")
    choice_source = a if isinstance(a, AbstractArray) else AbstractArray(shape=(1,), dtype="int64")
    return (
        AbstractArray(shape=source.shape, dtype=source.dtype),
        AbstractArray(shape=choice_source.shape, dtype=choice_source.dtype),
    )


@register_atom(witness_np_rand)  # type: ignore[untyped-decorator]
def rand(
    *dims: int | Sequence[int],
    size: ShapeLike | None = None,
    seed: SeedLike = None,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Return random values in a given shape over [0, 1)."""
    if len(dims) > 1 and not all(isinstance(dim, int) for dim in dims):
        raise TypeError("Multiple dimension arguments must all be ints")
    resolved_size = _resolve_rand_size(dims, size)
    gen = _resolve_generator(seed=seed, rng=rng)
    if gen is None:
        if resolved_size is None:
            return np.random.rand()
        if isinstance(resolved_size, int):
            return np.random.rand(resolved_size)
        return np.random.rand(*resolved_size)
    return gen.random(resolved_size)


@register_atom(witness_np_uniform)  # type: ignore[untyped-decorator]
@icontract.require(
    lambda low, high: bool(np.all(np.asarray(low) <= np.asarray(high))),
    "low must be less than or equal to high",
)
@icontract.require(
    lambda seed, rng: seed is None or rng is None or (isinstance(seed, np.random.Generator) and seed is rng),
    "Provide at most one of seed/rng unless they refer to the same Generator",
)
@icontract.ensure(
    lambda result, low, high, size: (
        isinstance(result, np.ndarray) and result.shape == expected_shape
        if (expected_shape := _uniform_result_shape(low, high, size)) is not None
        else isinstance(result, (float, np.floating))
    ),
    "Result shape must match requested size",
)
def uniform(
    low: ArrayLikeFloat = 0.0,
    high: ArrayLikeFloat = 1.0,
    size: ShapeLike | None = None,
    seed: SeedLike = None,
    rng: np.random.Generator | None = None,
) -> float | np.floating | np.ndarray:
    """Draw samples from a uniform distribution."""
    gen = _resolve_generator(seed=seed, rng=rng)
    if gen is None:
        return np.random.uniform(low, high, size)
    return gen.uniform(low, high, size)


@register_atom(witness_np_default_rng)  # type: ignore[untyped-decorator]
@icontract.require(
    lambda seed: seed is None
    or isinstance(seed, (int, np.random.SeedSequence, np.random.BitGenerator, np.random.Generator))
    or (isinstance(seed, Sequence) and not isinstance(seed, str)),
    "Invalid seed type",
)
@icontract.ensure(lambda result: isinstance(result, np.random.Generator), "Result must be a numpy Generator")
def default_rng(seed: SeedLike = None) -> np.random.Generator:
    """Construct a new NumPy Generator."""
    return np.random.default_rng(seed)


@register_atom(witness_continuous_multivariate_sampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda mean: mean is not None, "mean cannot be None")
@icontract.require(lambda cov: cov is not None, "cov cannot be None")
@icontract.require(lambda alpha: alpha is not None, "alpha cannot be None")
@icontract.require(lambda tol: isinstance(tol, (float, int, np.number)), "tol must be numeric")
@icontract.ensure(lambda result: all(r is not None for r in result), "Continuous multivariate sampler outputs must not be None")
def continuous_multivariate_sampler(
    mean: np.ndarray,
    cov: np.ndarray,
    alpha: np.ndarray,
    size: int | tuple[int, ...] | None = None,
    check_valid: str = "warn",
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw paired multivariate-normal and Dirichlet samples."""
    mvn_samples = np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol)
    dirichlet_samples = np.random.dirichlet(alpha, size=size)
    return mvn_samples, dirichlet_samples


@register_atom(witness_discrete_event_sampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda pvals: pvals is not None, "pvals cannot be None")
@icontract.ensure(lambda result: result is not None, "Discrete event sampler output must not be None")
def discrete_event_sampler(
    n: int,
    pvals: np.ndarray,
    size: int | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Draw count vectors from a multinomial distribution."""
    return np.random.multinomial(n, pvals, size=size)


@register_atom(witness_combinatorics_sampler)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "x cannot be None")
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "Combinatorics sampler outputs must not be None")
def combinatorics_sampler(
    x: int | np.ndarray,
    a: int | np.ndarray,
    size: int | tuple[int, ...] | None = None,
    replace: bool = True,
    p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a random permutation together with a random choice draw."""
    return np.random.permutation(x), np.random.choice(a, size=size, replace=replace, p=p)


continuousmultivariatesampler = continuous_multivariate_sampler
discreteeventsampler = discrete_event_sampler
combinatoricssampler = combinatorics_sampler


__all__ = [
    "rand",
    "uniform",
    "default_rng",
    "continuous_multivariate_sampler",
    "discrete_event_sampler",
    "combinatorics_sampler",
    "continuousmultivariatesampler",
    "discreteeventsampler",
    "combinatoricssampler",
]

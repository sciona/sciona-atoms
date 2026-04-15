"""NumPy search-and-sort wrappers sourced from the legacy v2 family."""

from __future__ import annotations

from typing import Sequence

import icontract
import numpy as np
from sciona.ghost.abstract import AbstractArray, AbstractScalar
from sciona.ghost.registry import register_atom


def witness_binary_search_insertion(
    a: AbstractArray,
    v: AbstractArray,
    side: AbstractScalar | None = None,
    sorter: AbstractArray | None = None,
) -> AbstractArray:
    return AbstractArray(shape=v.shape, dtype="int64", min_val=0.0)


def witness_lexicographic_indirect_sort(
    keys: AbstractArray,
    axis: AbstractScalar | None = None,
) -> AbstractArray:
    if not keys.shape:
        return AbstractArray(shape=(1,), dtype="int64", min_val=0.0)
    return AbstractArray(shape=(keys.shape[-1],), dtype="int64", min_val=0.0)


def witness_partial_sort_partition(
    a: AbstractArray,
    kth: AbstractScalar | AbstractArray,
    axis: AbstractScalar | None = None,
    kind: AbstractScalar | None = None,
    order: AbstractScalar | AbstractArray | None = None,
) -> AbstractArray:
    return AbstractArray(shape=a.shape, dtype=a.dtype)


@register_atom(witness_binary_search_insertion)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda v: v is not None, "v cannot be None")
@icontract.require(lambda side: side in {"left", "right"}, "side must be 'left' or 'right'")
@icontract.ensure(lambda result: result is not None, "Binary search insertion output must not be None")
def binary_search_insertion(
    a: np.ndarray,
    v: np.ndarray | int | float,
    side: str = "left",
    sorter: np.ndarray | None = None,
) -> np.ndarray:
    """Locate insertion points for values into a sorted array."""
    return np.searchsorted(a, v, side=side, sorter=sorter)


@register_atom(witness_lexicographic_indirect_sort)  # type: ignore[untyped-decorator]
@icontract.require(lambda keys: keys is not None, "keys cannot be None")
@icontract.ensure(lambda result: result is not None, "Lexicographic indirect sort output must not be None")
def lexicographic_indirect_sort(keys: Sequence[np.ndarray]) -> np.ndarray:
    """Return an indirect lexicographic sort permutation for the given keys."""
    return np.lexsort(keys)


@register_atom(witness_partial_sort_partition)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "a cannot be None")
@icontract.require(lambda kth: kth is not None, "kth cannot be None")
@icontract.ensure(lambda result: all(r is not None for r in result), "Partial sort partition outputs must not be None")
def partial_sort_partition(
    a: np.ndarray,
    kth: int | Sequence[int],
    axis: int | None = -1,
    kind: str = "introselect",
    order: str | list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a partitioned array together with the corresponding argpartition indices."""
    return (
        np.partition(a, kth, axis=axis, kind=kind, order=order),
        np.argpartition(a, kth, axis=axis, kind=kind, order=order),
    )


# Compatibility aliases preserving the legacy source family spellings.
binarysearchinsertion = binary_search_insertion
lexicographicindirectsort = lexicographic_indirect_sort
partialsortpartition = partial_sort_partition


__all__ = [
    "binary_search_insertion",
    "lexicographic_indirect_sort",
    "partial_sort_partition",
    "binarysearchinsertion",
    "lexicographicindirectsort",
    "partialsortpartition",
]

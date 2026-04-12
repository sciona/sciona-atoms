"""NumPy linear-algebra wrappers."""

from __future__ import annotations

import icontract
import numpy as np
from ageoa.ghost.registry import register_atom
from ageoa.numpy.witnesses import (
    witness_np_linalg_det,
    witness_np_linalg_inv,
    witness_np_linalg_norm,
    witness_np_linalg_solve,
)

ArrayLike = np.ndarray | list[object] | tuple[object, ...]


def _is_square_2d(a: ArrayLike) -> bool:
    a_arr = np.asarray(a)
    return a_arr.ndim == 2 and a_arr.shape[0] == a_arr.shape[1]


def _is_square_at_least_2d(a: ArrayLike) -> bool:
    a_arr = np.asarray(a)
    return a_arr.ndim >= 2 and a_arr.shape[-1] == a_arr.shape[-2]


@register_atom(witness_np_linalg_solve)  # type: ignore[untyped-decorator]
@icontract.require(lambda a, b: np.asarray(a).ndim == 2, "a must be a 2D matrix")
@icontract.require(lambda a, b: _is_square_2d(a), "a must be square")
@icontract.require(
    lambda a, b: np.asarray(a).shape[0] == np.asarray(b).shape[0],
    "Dimensions of a and b must match",
)
@icontract.ensure(lambda result, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve a linear system."""
    return np.linalg.solve(a, b)


@register_atom(witness_np_linalg_inv)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result.shape == np.asarray(a).shape, "Inverse has same shape as input")
def inv(a: np.ndarray) -> np.ndarray:
    """Compute the inverse of a matrix."""
    return np.linalg.inv(a)


@register_atom(witness_np_linalg_det)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: np.asarray(a).ndim >= 2, "a must have at least 2 dimensions")
@icontract.require(
    lambda a: _is_square_at_least_2d(a),
    "The last two dimensions of a must be square",
)
@icontract.ensure(lambda result: result is not None, "Determinant must not be None")
def det(a: np.ndarray) -> np.floating | np.ndarray:
    """Compute the determinant of an array."""
    return np.linalg.det(a)


@register_atom(witness_np_linalg_norm)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(lambda result: np.all(np.asarray(result) >= 0), "Norm must be non-negative")
def norm(
    x: ArrayLike,
    ord: int | float | str | None = None,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> float | np.floating | np.ndarray:
    """Return a matrix or vector norm."""
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


__all__ = ["solve", "inv", "det", "norm"]

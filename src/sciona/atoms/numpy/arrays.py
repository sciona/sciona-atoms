"""NumPy array-construction and shape-manipulation wrappers."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import icontract
import numpy as np
from sciona.ghost.registry import register_atom
from sciona.atoms.numpy.witnesses import (
    witness_np_array,
    witness_np_dot,
    witness_np_reshape,
    witness_np_vstack,
    witness_np_zeros,
)

DTypeLike = np.dtype | type | str | None
ShapeLike = Integral | Sequence[Integral]
ScalarLike = float | int | complex | bool | np.number
ArrayLike = np.ndarray | Sequence[ScalarLike] | ScalarLike


def _shape_tuple(shape: ShapeLike) -> tuple[int, ...]:
    if isinstance(shape, Integral):
        return (int(shape),)
    return tuple(int(dim) for dim in shape)


def _check_dot_dims(a: ArrayLike, b: ArrayLike) -> bool:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.ndim == 0 or b_arr.ndim == 0:
        return True
    if a_arr.ndim == 1 and b_arr.ndim == 1:
        return a_arr.shape[0] == b_arr.shape[0]
    if b_arr.ndim == 1:
        return a_arr.shape[-1] == b_arr.shape[0]
    return a_arr.shape[-1] == b_arr.shape[-2]


@register_atom(witness_np_array)  # type: ignore[untyped-decorator]
@icontract.require(lambda object: object is not None, "Object must not be None")
@icontract.ensure(lambda result: isinstance(result, np.ndarray), "Result must be a numpy array")
def array(
    object: ArrayLike,  # noqa: A002
    dtype: DTypeLike = None,
    *,
    copy: bool | None = True,
    order: str = "K",
    subok: bool = False,
    ndmin: int = 0,
    ndmax: int = 0,
    like: ArrayLike | None = None,
) -> np.ndarray:
    """Create a NumPy array."""
    return np.array(
        object,
        dtype=dtype,
        copy=copy,
        order=order,
        subok=subok,
        ndmin=ndmin,
        ndmax=ndmax,
        like=like,
    )


@register_atom(witness_np_zeros)  # type: ignore[untyped-decorator]
@icontract.require(
    lambda shape: isinstance(shape, (Integral, tuple, list)),
    "Shape must be an int or a sequence of ints",
)
@icontract.ensure(
    lambda result, shape: result.shape == _shape_tuple(shape),
    "Result shape must match requested shape",
)
def zeros(
    shape: ShapeLike,
    dtype: DTypeLike = None,
    order: str = "C",
    *,
    device: str | None = None,
    like: ArrayLike | None = None,
) -> np.ndarray:
    """Return a new array of given shape and type, filled with zeros."""
    return np.zeros(shape, dtype=dtype, order=order, device=device, like=like)


@register_atom(witness_np_dot)  # type: ignore[untyped-decorator]
@icontract.require(
    lambda a, b: _check_dot_dims(a, b),
    "Dimensions of a and b must be compatible for dot product",
)
@icontract.ensure(lambda result: result is not None, "Result must not be None")
def dot(a: ArrayLike, b: ArrayLike, out: np.ndarray | None = None) -> np.ndarray | np.number:
    """Return the dot product of two arrays."""
    return np.dot(a, b, out=out)


@register_atom(witness_np_vstack)  # type: ignore[untyped-decorator]
@icontract.require(lambda tup: len(tup) > 0, "Sequence of arrays must not be empty")
@icontract.ensure(
    lambda result, tup: result.shape[0]
    == sum(np.asarray(x).shape[0] if np.asarray(x).ndim > 1 else 1 for x in tup),
    "Result leading dimension must match sum of input leading dimensions",
)
def vstack(
    tup: Sequence[ArrayLike],
    *,
    dtype: DTypeLike = None,
    casting: str = "same_kind",
) -> np.ndarray:
    """Stack arrays vertically."""
    return np.vstack(tup, dtype=dtype, casting=casting)


@register_atom(witness_np_reshape)  # type: ignore[untyped-decorator]
@icontract.require(lambda a: a is not None, "Array must not be None")
@icontract.ensure(lambda result, a: result.size == np.asarray(a).size, "Result size must match original array size")
def reshape(
    a: ArrayLike,
    /,
    shape: ShapeLike,
    order: str = "C",
    *,
    copy: bool | None = None,
) -> np.ndarray:
    """Give an array a new shape without changing its data."""
    return np.reshape(a, shape, order=order, copy=copy)


__all__ = ["array", "zeros", "dot", "vstack", "reshape"]

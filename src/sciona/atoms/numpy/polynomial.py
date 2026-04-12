"""NumPy polynomial wrappers."""

from __future__ import annotations

import icontract
import numpy as np
import numpy.polynomial.polynomial as poly
from ageoa.ghost.registry import register_atom
from ageoa.numpy.witnesses import (
    witness_np_polyadd,
    witness_np_polyder,
    witness_np_polyfit,
    witness_np_polyint,
    witness_np_polymul,
    witness_np_polyroots,
    witness_np_polyval,
)

ArrayLike = np.ndarray | list[object] | tuple[object, ...]
CoefficientLike = np.ndarray | list[object] | tuple[object, ...]
PolyValueLike = np.ndarray | list[object] | tuple[object, ...] | float | int | complex | np.number


@register_atom(witness_np_polyval)  # type: ignore[untyped-decorator]
@icontract.require(lambda c, x: c is not None and x is not None, "Coefficients and x must not be None")
@icontract.require(lambda c: len(np.asarray(c)) > 0, "Coefficients must not be empty")
@icontract.ensure(lambda result, x: np.asarray(result).shape == np.asarray(x).shape, "Result shape must match x shape")
def polyval(x: PolyValueLike, c: CoefficientLike) -> np.ndarray | np.number:
    """Evaluate a polynomial at points x."""
    return poly.polyval(x, c)


@register_atom(witness_np_polyfit)  # type: ignore[untyped-decorator]
@icontract.require(lambda x, y, deg: len(np.asarray(x)) == len(np.asarray(y)), "x and y must have the same length")
@icontract.require(lambda deg: deg >= 0, "Degree must be non-negative")
@icontract.ensure(lambda result, deg: len(result) == deg + 1, "Result must have deg + 1 coefficients")
def polyfit(x: ArrayLike, y: ArrayLike, deg: int) -> np.ndarray:
    """Fit a polynomial to data in the least-squares sense."""
    return poly.polyfit(x, y, deg)


@register_atom(witness_np_polyder)  # type: ignore[untyped-decorator]
@icontract.require(lambda c: len(np.asarray(c)) > 0, "Coefficients must not be empty")
@icontract.ensure(lambda result, c, m: len(result) == max(1, len(c) - m), "Result length must be correct")
def polyder(c: CoefficientLike, m: int = 1) -> np.ndarray:
    """Differentiate a polynomial."""
    return poly.polyder(c, m=m)


@register_atom(witness_np_polyint)  # type: ignore[untyped-decorator]
@icontract.require(lambda c: len(np.asarray(c)) > 0, "Coefficients must not be empty")
@icontract.require(lambda m, k: (len(k) if np.iterable(k) else 1) <= m, "Too many integration constants")
@icontract.ensure(lambda result, c, m: len(result) == len(c) + m, "Result length must be correct")
def polyint(c: CoefficientLike, m: int = 1, k: ArrayLike | float = 0) -> np.ndarray:
    """Integrate a polynomial."""
    return poly.polyint(c, m=m, k=k)


@register_atom(witness_np_polyadd)  # type: ignore[untyped-decorator]
@icontract.require(
    lambda c1, c2: len(np.asarray(c1)) > 0 and len(np.asarray(c2)) > 0,
    "Coefficients must not be empty",
)
@icontract.ensure(
    lambda result, c1, c2: len(result) == max(len(c1), len(c2)),
    "Result length must match maximum of input lengths",
)
def polyadd(c1: CoefficientLike, c2: CoefficientLike) -> np.ndarray:
    """Add one polynomial to another."""
    return poly.polyadd(c1, c2)


@register_atom(witness_np_polymul)  # type: ignore[untyped-decorator]
@icontract.require(
    lambda c1, c2: len(np.asarray(c1)) > 0 and len(np.asarray(c2)) > 0,
    "Coefficients must not be empty",
)
@icontract.ensure(
    lambda result, c1, c2: len(result) == len(c1) + len(c2) - 1 if len(c1) > 0 and len(c2) > 0 else 0,
    "Result length must match product of input lengths",
)
def polymul(c1: CoefficientLike, c2: CoefficientLike) -> np.ndarray:
    """Multiply one polynomial by another."""
    return poly.polymul(c1, c2)


@register_atom(witness_np_polyroots)  # type: ignore[untyped-decorator]
@icontract.require(lambda c: len(np.asarray(c)) >= 2, "Polynomial must have at least degree 1 to have roots")
@icontract.ensure(lambda result, c: len(result) == len(c) - 1, "Number of roots must match polynomial degree")
def polyroots(c: CoefficientLike) -> np.ndarray:
    """Compute the roots of a polynomial."""
    return poly.polyroots(c)


__all__ = ["polyval", "polyfit", "polyder", "polyint", "polyadd", "polymul", "polyroots"]

"""NumPy emath wrappers."""

from __future__ import annotations

import os

import icontract
import numpy as np
from sciona.ghost.registry import register_atom
from sciona.atoms.numpy.witnesses import (
    witness_np_emath_log,
    witness_np_emath_log10,
    witness_np_emath_logn,
    witness_np_emath_power,
    witness_np_emath_sqrt,
)

_SLOW_CHECKS = os.environ.get("SCIONA_SLOW_CHECKS", "0") == "1"

Numeric = float | int | complex | np.number
ArrayLike = np.ndarray | list[object] | tuple[object, ...] | Numeric
EmathResult = np.ndarray | np.number | complex | float


@register_atom(witness_np_emath_sqrt)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(
    lambda result, x: np.allclose(np.square(result), x),
    "Result squared must be approximately x",
    enabled=_SLOW_CHECKS,
)
def sqrt(x: ArrayLike) -> EmathResult:
    """Compute square root with NumPy scimath complex promotion for negative real inputs."""
    return np.emath.sqrt(x)


@register_atom(witness_np_emath_log)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(
    lambda result, x: np.allclose(np.exp(result), x),
    "Exp of result must be approximately x",
    enabled=_SLOW_CHECKS,
)
def log(x: ArrayLike) -> EmathResult:
    """Compute the principal natural logarithm, promoting negative real inputs to complex."""
    return np.emath.log(x)


@register_atom(witness_np_emath_log10)  # type: ignore[untyped-decorator]
@icontract.require(lambda x: x is not None, "Input must not be None")
@icontract.ensure(
    lambda result, x: np.allclose(np.power(10, result), x),
    "10 to the power of result must be approximately x",
    enabled=_SLOW_CHECKS,
)
def log10(x: ArrayLike) -> EmathResult:
    """Compute the principal base-10 logarithm, promoting negative real inputs to complex."""
    return np.emath.log10(x)


@register_atom(witness_np_emath_logn)  # type: ignore[untyped-decorator]
@icontract.require(lambda n, x: n is not None and x is not None, "Base n and value x must not be None")
@icontract.ensure(lambda result: result is not None, "Result must not be None")
def logn(n: Numeric, x: ArrayLike) -> EmathResult:
    """Compute the principal logarithm base n using NumPy scimath domain handling."""
    return np.emath.logn(n, x)


@register_atom(witness_np_emath_power)  # type: ignore[untyped-decorator]
@icontract.require(lambda x, p: x is not None and p is not None, "Input x and power p must not be None")
@icontract.ensure(lambda result: result is not None, "Result must not be None")
def power(x: ArrayLike, p: ArrayLike) -> EmathResult:
    """Return x to the power p, promoting negative real bases to complex."""
    return np.emath.power(x, p)


__all__ = ["sqrt", "log", "log10", "logn", "power"]

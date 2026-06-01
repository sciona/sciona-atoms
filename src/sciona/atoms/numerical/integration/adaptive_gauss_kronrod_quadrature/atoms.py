from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_evaluate_gauss_kronrod_15,
    witness_adaptive_subdivision_loop,
)

@register_atom(witness_evaluate_gauss_kronrod_15, name="evaluate_gauss_kronrod_15")
@icontract.require(lambda func, a, b: a < b, "Precondition failed: a < b")
@icontract.ensure(lambda result, func, a, b: error_estimate >= 0, "Postcondition failed: error_estimate >= 0")
def evaluate_gauss_kronrod_15(func: float, a: float, b: float) -> float:
    """Evaluate Gauss 7-point and Kronrod 15-point rules on a single interval to estimate the integral and its localized error.

    Args:
        func: Callable[[float], float]
        a: float
        b: float

    Returns:
        integral_estimate: float
    """
    import scipy.integrate
    return scipy.integrate.quad(func=func, a=a, b=b) # type: ignore

@register_atom(witness_adaptive_subdivision_loop, name="adaptive_subdivision_loop")
@icontract.require(lambda func, a, b, epsabs, epsrel, limit: limit > 0, "Precondition failed: limit > 0")
@icontract.ensure(lambda result, func, a, b, epsabs, epsrel, limit: total_error >= 0, "Postcondition failed: total_error >= 0")
def adaptive_subdivision_loop(func: float, a: float, b: float, epsabs: float, epsrel: float, limit: int) -> float:
    """Execute the adaptive interval subdivision loop using a priority queue ranked by error estimate.

    Args:
        func: Callable[[float], float]
        a: float
        b: float
        epsabs: float
        epsrel: float
        limit: int

    Returns:
        result: float
    """
    import scipy.integrate
    return scipy.integrate.quad(func=func, a=a, b=b, epsabs=epsabs, epsrel=epsrel, limit=limit) # type: ignore


from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_generate_knot_vector,
    witness_solve_bspline_coefficients,
    witness_evaluate_bspline_cox_de_boor,
)

@register_atom(witness_generate_knot_vector, name="generate_knot_vector")
@icontract.require(lambda x, k: len(x) >= k + 1, "Precondition failed: len(x) >= k + 1")
@icontract.require(lambda x, k: np.all(np.diff(x) > 0), "Precondition failed: np.all(np.diff(x) > 0)")
@icontract.ensure(lambda result, x, k: len(knots) == len(x) + k + 1, "Postcondition failed: len(knots) == len(x) + k + 1")
def generate_knot_vector(x: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Generate clamped or uniform knot vector for B-splines.

    Args:
        x: NDArray[np.float64]
        k: k >= 0

    Returns:
        knots: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.make_interp_spline(x=x, k=k) # type: ignore

@register_atom(witness_solve_bspline_coefficients, name="solve_bspline_coefficients")
@icontract.require(lambda x, y, knots, k: len(knots) == len(x) + k + 1, "Precondition failed: len(knots) == len(x) + k + 1")
@icontract.ensure(lambda result, x, y, knots, k: len(coefficients) == len(x), "Postcondition failed: len(coefficients) == len(x)")
def solve_bspline_coefficients(x: NDArray[np.float64], y: NDArray[np.float64], knots: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """Solve the linear interpolation/least-squares system for B-spline coefficients.

    Args:
        x: NDArray[np.float64]
        y: NDArray[np.float64]
        knots: NDArray[np.float64]
        k: int

    Returns:
        coefficients: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.make_interp_spline(x=x, y=y, knots=knots, k=k) # type: ignore

@register_atom(witness_evaluate_bspline_cox_de_boor, name="evaluate_bspline_cox_de_boor")
@icontract.require(lambda knots, coefficients, k, x_eval: len(knots) == len(coefficients) + k + 1, "Precondition failed: len(knots) == len(coefficients) + k + 1")
@icontract.ensure(lambda result, knots, coefficients, k, x_eval: len(result) == len(x_eval), "Postcondition failed: len(result) == len(x_eval)")
def evaluate_bspline_cox_de_boor(knots: NDArray[np.float64], coefficients: NDArray[np.float64], k: int, x_eval: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate B-spline curve at specified evaluation coordinates using recursive Cox-de Boor algorithm.

    Args:
        knots: NDArray[np.float64]
        coefficients: NDArray[np.float64]
        k: int
        x_eval: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.BSpline(knots=knots, coefficients=coefficients, k=k, x_eval=x_eval) # type: ignore


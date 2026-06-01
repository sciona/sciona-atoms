from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_assemble_cubic_spline_tridiagonal,
    witness_solve_tridiagonal_thomas,
    witness_compute_cubic_spline_coefficients,
    witness_evaluate_piecewise_polynomial,
)

@register_atom(witness_assemble_cubic_spline_tridiagonal, name="assemble_cubic_spline_tridiagonal")
@icontract.require(lambda x, y, bc_type: len(x) >= 3, "Precondition failed: len(x) >= 3")
@icontract.require(lambda x, y, bc_type: np.all(np.diff(x) > 0), "Precondition failed: np.all(np.diff(x) > 0)")
@icontract.ensure(lambda result, x, y, bc_type: ab.shape[0] == 3, "Postcondition failed: ab.shape[0] == 3")
@icontract.ensure(lambda result, x, y, bc_type: ab.shape[1] == len(x), "Postcondition failed: ab.shape[1] == len(x)")
@icontract.ensure(lambda result, x, y, bc_type: len(rhs) == len(x), "Postcondition failed: len(rhs) == len(x)")
def assemble_cubic_spline_tridiagonal(x: NDArray[np.float64], y: NDArray[np.float64], bc_type: str) -> NDArray[np.float64]:
    """Build the tridiagonal system of linear equations for the spline derivatives based on boundary conditions.

    Args:
        x: NDArray[np.float64]
        y: NDArray[np.float64]
        bc_type: Must be 'natural', 'clamped', or 'not-a-knot'

    Returns:
        ab: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.CubicSpline(x=x, y=y, bc_type=bc_type) # type: ignore

@register_atom(witness_solve_tridiagonal_thomas, name="solve_tridiagonal_thomas")
@icontract.require(lambda ab, rhs: ab.ndim == 2, "Precondition failed: ab.ndim == 2")
@icontract.require(lambda ab, rhs: ab.shape[0] == 3, "Precondition failed: ab.shape[0] == 3")
@icontract.ensure(lambda result, ab, rhs: len(derivatives) == len(rhs), "Postcondition failed: len(derivatives) == len(rhs)")
def solve_tridiagonal_thomas(ab: NDArray[np.float64], rhs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the tridiagonal system using Thomas algorithm.

    Args:
        ab: NDArray[np.float64]
        rhs: NDArray[np.float64]

    Returns:
        derivatives: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.solve_banded(ab=ab, rhs=rhs) # type: ignore

@register_atom(witness_compute_cubic_spline_coefficients, name="compute_cubic_spline_coefficients")
@icontract.require(lambda x, y, derivatives: len(derivatives) == len(x), "Precondition failed: len(derivatives) == len(x)")
@icontract.ensure(lambda result, x, y, derivatives: c.shape == (4, len(x)-1), "Postcondition failed: c.shape == (4, len(x)-1)")
def compute_cubic_spline_coefficients(x: NDArray[np.float64], y: NDArray[np.float64], derivatives: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert knot values and derivatives into piecewise cubic polynomial coefficients.

    Args:
        x: NDArray[np.float64]
        y: NDArray[np.float64]
        derivatives: NDArray[np.float64]

    Returns:
        c: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.CubicSpline(x=x, y=y, derivatives=derivatives) # type: ignore

@register_atom(witness_evaluate_piecewise_polynomial, name="evaluate_piecewise_polynomial")
@icontract.require(lambda x, c, x_eval: c.shape[1] == len(x) - 1, "Precondition failed: c.shape[1] == len(x) - 1")
@icontract.ensure(lambda result, x, c, x_eval: len(result) == len(x_eval), "Postcondition failed: len(result) == len(x_eval)")
def evaluate_piecewise_polynomial(x: NDArray[np.float64], c: NDArray[np.float64], x_eval: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate local polynomial representation at requested points.

    Args:
        x: NDArray[np.float64]
        c: NDArray[np.float64]
        x_eval: NDArray[np.float64]

    Returns:
        result: NDArray[np.float64]
    """
    import scipy.interpolate
    return scipy.interpolate.PPoly(x=x, c=c, x_eval=x_eval) # type: ignore


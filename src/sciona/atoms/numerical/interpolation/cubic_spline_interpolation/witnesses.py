from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_assemble_cubic_spline_tridiagonal(x: AbstractArray, y: AbstractArray, bc_type: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for assemble_cubic_spline_tridiagonal."""
    _ = (x, y, bc_type)
    return AbstractArray(shape=x.shape, dtype=x.dtype)

def witness_solve_tridiagonal_thomas(ab: AbstractArray, rhs: AbstractArray) -> AbstractArray:
    """Ghost witness for solve_tridiagonal_thomas."""
    _ = (ab, rhs)
    return AbstractArray(shape=ab.shape, dtype=ab.dtype)

def witness_compute_cubic_spline_coefficients(x: AbstractArray, y: AbstractArray, derivatives: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_cubic_spline_coefficients."""
    _ = (x, y, derivatives)
    return AbstractArray(shape=x.shape, dtype=x.dtype)

def witness_evaluate_piecewise_polynomial(x: AbstractArray, c: AbstractArray, x_eval: AbstractArray) -> AbstractArray:
    """Ghost witness for evaluate_piecewise_polynomial."""
    _ = (x, c, x_eval)
    return AbstractArray(shape=x.shape, dtype=x.dtype)


from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_generate_knot_vector(x: AbstractArray, k: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for generate_knot_vector."""
    _ = (x, k)
    return AbstractArray(shape=x.shape, dtype=x.dtype)

def witness_solve_bspline_coefficients(x: AbstractArray, y: AbstractArray, knots: AbstractArray, k: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for solve_bspline_coefficients."""
    _ = (x, y, knots, k)
    return AbstractArray(shape=x.shape, dtype=x.dtype)

def witness_evaluate_bspline_cox_de_boor(knots: AbstractArray, coefficients: AbstractArray, k: AbstractScalar | int, x_eval: AbstractArray) -> AbstractArray:
    """Ghost witness for evaluate_bspline_cox_de_boor."""
    _ = (knots, coefficients, k, x_eval)
    return AbstractArray(shape=knots.shape, dtype=knots.dtype)


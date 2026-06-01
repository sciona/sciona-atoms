from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_get_butcher_tableau(method: AbstractScalar | str) -> AbstractArray:
    """Ghost witness for get_butcher_tableau."""
    _ = (method)
    return AbstractArray(shape=(), dtype="float64")

def witness_compute_rk_single_step(fun: AbstractArray, t: AbstractScalar | float, y: AbstractArray, h: AbstractScalar | float, A: AbstractArray, B: AbstractArray, B_star: AbstractArray, C: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_rk_single_step."""
    _ = (fun, t, y, h, A, B, B_star, C)
    return AbstractArray(shape=fun.shape, dtype=fun.dtype)

def witness_evaluate_step_acceptability(y: AbstractArray, y_new: AbstractArray, error_vector: AbstractArray, atol: AbstractScalar | float, rtol: AbstractScalar | float) -> AbstractScalar:
    """Ghost witness for evaluate_step_acceptability."""
    _ = (y, y_new, error_vector, atol, rtol)
    return AbstractScalar(dtype="float64")


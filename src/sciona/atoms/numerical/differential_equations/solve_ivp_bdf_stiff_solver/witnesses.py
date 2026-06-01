from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_bdf_predictor(history: AbstractArray, order: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for bdf_predictor."""
    _ = (history, order)
    return AbstractArray(shape=history.shape, dtype=history.dtype)

def witness_bdf_newton_corrector(fun: AbstractArray, t_new: AbstractScalar | float, y_predict: AbstractArray, h: AbstractScalar | float, beta: AbstractScalar | float, jacobian: AbstractArray, rhs_constant: AbstractArray, atol: AbstractScalar | float, rtol: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for bdf_newton_corrector."""
    _ = (fun, t_new, y_predict, h, beta, jacobian, rhs_constant, atol, rtol)
    return AbstractArray(shape=fun.shape, dtype=fun.dtype)


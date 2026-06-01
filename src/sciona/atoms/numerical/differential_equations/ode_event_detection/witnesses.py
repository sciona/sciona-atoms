from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_evaluate_and_check_event_crossings(event_func: AbstractArray, t_prev: AbstractScalar | float, y_prev: AbstractArray, t_curr: AbstractScalar | float, y_curr: AbstractArray, direction: AbstractArray) -> AbstractArray:
    """Ghost witness for evaluate_and_check_event_crossings."""
    _ = (event_func, t_prev, y_prev, t_curr, y_curr, direction)
    return AbstractArray(shape=event_func.shape, dtype=event_func.dtype)

def witness_formulate_event_brackets(t_prev: AbstractScalar | float, t_curr: AbstractScalar | float, active_mask: AbstractArray) -> AbstractScalar:
    """Ghost witness for formulate_event_brackets."""
    _ = (t_prev, t_curr, active_mask)
    return AbstractScalar(dtype="float64")


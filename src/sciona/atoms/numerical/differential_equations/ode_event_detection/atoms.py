from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_evaluate_and_check_event_crossings,
    witness_formulate_event_brackets,
)

@register_atom(witness_evaluate_and_check_event_crossings, name="evaluate_and_check_event_crossings")
@icontract.require(lambda event_func, t_prev, y_prev, t_curr, y_curr, direction: direction.ndim == 1, "Precondition failed: direction.ndim == 1")
@icontract.ensure(lambda result, event_func, t_prev, y_prev, t_curr, y_curr, direction: len(active_mask) == len(direction), "Postcondition failed: len(active_mask) == len(direction)")
def evaluate_and_check_event_crossings(event_func: Callable[[float, NDArray[np.float64]], NDArray[np.float64]], t_prev: float, y_prev: NDArray[np.float64], t_curr: float, y_curr: NDArray[np.float64], direction: NDArray[np.int64]) -> NDArray[np.bool_]:
    """Evaluate event function at step endpoints and identify which indices exhibit valid zero crossings.

    Args:
        event_func: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
        t_prev: float
        y_prev: NDArray[np.float64]
        t_curr: float
        y_curr: NDArray[np.float64]
        direction: NDArray[np.int64]

    Returns:
        active_mask: NDArray[np.bool_]
    """
    import scipy.integrate._ivp.common
    return scipy.integrate._ivp.common.find_active_events(event_func=event_func, t_prev=t_prev, y_prev=y_prev, t_curr=t_curr, y_curr=y_curr, direction=direction) # type: ignore

@register_atom(witness_formulate_event_brackets, name="formulate_event_brackets")
@icontract.require(lambda t_prev, t_curr, active_mask: t_prev is not None, "Precondition failed: t_prev is not None")
@icontract.ensure(lambda result, t_prev, t_curr, active_mask: len(brackets) == np.sum(active_mask), "Postcondition failed: len(brackets) == np.sum(active_mask)")
def formulate_event_brackets(t_prev: float, t_curr: float, active_mask: NDArray[np.bool_]) -> float:
    """Generate bounding intervals for triggered events.

    Args:
        t_prev: float
        t_curr: float
        active_mask: NDArray[np.bool_]

    Returns:
        brackets: list[tuple[float, float]]
    """
    import scipy.integrate._ivp.common
    return scipy.integrate._ivp.common.find_active_events(t_prev=t_prev, t_curr=t_curr, active_mask=active_mask) # type: ignore


from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_select_kde_bandwidth(data: AbstractArray, bw_method: AbstractScalar | str) -> AbstractScalar:
    """Ghost witness for select_kde_bandwidth."""
    _ = (data, bw_method)
    return AbstractScalar(dtype="float64")

def witness_evaluate_kde_density(data: AbstractArray, eval_points: AbstractArray, factor: AbstractScalar | float) -> AbstractArray:
    """Ghost witness for evaluate_kde_density."""
    _ = (data, eval_points, factor)
    return AbstractArray(shape=data.shape, dtype=data.dtype)


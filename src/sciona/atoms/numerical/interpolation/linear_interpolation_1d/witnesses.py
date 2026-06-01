from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_validate_and_sort_interpolation_grid(x_grid: AbstractArray, y_values: AbstractArray) -> AbstractArray:
    """Ghost witness for validate_and_sort_interpolation_grid."""
    _ = (x_grid, y_values)
    return AbstractArray(shape=x_grid.shape, dtype=x_grid.dtype)

def witness_search_interval_indices(grid: AbstractArray, points: AbstractArray) -> AbstractArray:
    """Ghost witness for search_interval_indices."""
    _ = (grid, points)
    return AbstractArray(shape=grid.shape, dtype=grid.dtype)

def witness_compute_linear_blend(grid: AbstractArray, values: AbstractArray, points: AbstractArray, indices: AbstractArray) -> AbstractArray:
    """Ghost witness for compute_linear_blend."""
    _ = (grid, values, points, indices)
    return AbstractArray(shape=grid.shape, dtype=grid.dtype)


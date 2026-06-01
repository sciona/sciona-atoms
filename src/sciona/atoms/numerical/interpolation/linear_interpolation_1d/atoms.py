from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_validate_and_sort_interpolation_grid,
    witness_search_interval_indices,
    witness_compute_linear_blend,
)

@register_atom(witness_validate_and_sort_interpolation_grid, name="validate_and_sort_interpolation_grid")
@icontract.require(lambda x_grid, y_values: x_grid.ndim == 1, "Precondition failed: x_grid.ndim == 1")
@icontract.require(lambda x_grid, y_values: len(x_grid) >= 2, "Precondition failed: len(x_grid) >= 2")
@icontract.require(lambda x_grid, y_values: len(x_grid) == len(y_values), "Precondition failed: len(x_grid) == len(y_values)")
@icontract.ensure(lambda result, x_grid, y_values: np.all(np.diff(sorted_x) > 0), "Postcondition failed: np.all(np.diff(sorted_x) > 0)")
def validate_and_sort_interpolation_grid(x_grid: NDArray[np.float64], y_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Verify and sort input grid coordinates to ensure strict monotonicity.

    Args:
        x_grid: NDArray[np.float64]
        y_values: NDArray[np.float64]

    Returns:
        sorted_x: NDArray[np.float64]
    """
    import numpy
    return numpy.sort(x_grid=x_grid, y_values=y_values) # type: ignore

@register_atom(witness_search_interval_indices, name="search_interval_indices")
@icontract.require(lambda grid, points: np.all(np.diff(grid) > 0), "Precondition failed: np.all(np.diff(grid) > 0)")
@icontract.ensure(lambda result, grid, points: np.all(indices >= 0), "Postcondition failed: np.all(indices >= 0)")
@icontract.ensure(lambda result, grid, points: np.all(indices < len(grid)), "Postcondition failed: np.all(indices < len(grid))")
def search_interval_indices(grid: NDArray[np.float64], points: NDArray[np.float64]) -> NDArray[np.int64]:
    """Identify grid intervals containing evaluation points using binary search.

    Args:
        grid: NDArray[np.float64]
        points: NDArray[np.float64]

    Returns:
        indices: NDArray[np.int64]
    """
    import numpy
    return numpy.searchsorted(grid=grid, points=points) # type: ignore

@register_atom(witness_compute_linear_blend, name="compute_linear_blend")
@icontract.require(lambda grid, values, points, indices: len(indices) == len(points), "Precondition failed: len(indices) == len(points)")
@icontract.ensure(lambda result, grid, values, points, indices: len(result) == len(points), "Postcondition failed: len(result) == len(points)")
def compute_linear_blend(grid: NDArray[np.float64], values: NDArray[np.float64], points: NDArray[np.float64], indices: NDArray[np.int64]) -> NDArray[np.float64]:
    """Calculate linear blend weights and evaluate final values.

    Args:
        grid: NDArray[np.float64]
        values: NDArray[np.float64]
        points: NDArray[np.float64]
        indices: NDArray[np.int64]

    Returns:
        result: NDArray[np.float64]
    """
    import numpy
    return numpy.interp(grid=grid, values=values, points=points, indices=indices) # type: ignore


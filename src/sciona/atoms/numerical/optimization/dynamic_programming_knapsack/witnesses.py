from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
from sciona.ghost.abstract import AbstractArray, AbstractScalar, AbstractSignal

def witness_fill_knapsack_table(values: AbstractArray, weights: AbstractArray, capacity: AbstractScalar | int) -> AbstractArray:
    """Ghost witness for fill_knapsack_table."""
    _ = (values, weights, capacity)
    return AbstractArray(shape=values.shape, dtype=values.dtype)

def witness_backtrack_knapsack_items(weights: AbstractArray, dp_table: AbstractArray) -> AbstractArray:
    """Ghost witness for backtrack_knapsack_items."""
    _ = (weights, dp_table)
    return AbstractArray(shape=weights.shape, dtype=weights.dtype)


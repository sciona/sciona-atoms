from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_fill_knapsack_table,
    witness_backtrack_knapsack_items,
)

@register_atom(witness_fill_knapsack_table, name="fill_knapsack_table")
@icontract.require(lambda values, weights, capacity: values.shape[0] == weights.shape[0], "Precondition failed: values.shape[0] == weights.shape[0]")
@icontract.require(lambda values, weights, capacity: capacity >= 0, "Precondition failed: capacity >= 0")
@icontract.ensure(lambda result, values, weights, capacity: result is not None, "Postcondition failed: result is not None")
def fill_knapsack_table(values: NDArray[np.float64], weights: NDArray[np.int64], capacity: int) -> NDArray[np.float64]:
    """Fill DP matrix based on item weights and capacities.

    Args:
        values: NDArray[np.float64]
        weights: NDArray[np.int64]
        capacity: int

    Returns:
        dp_table: NDArray[np.float64]
    """
    return needs_human_decision() # type: ignore

@register_atom(witness_backtrack_knapsack_items, name="backtrack_knapsack_items")
@icontract.require(lambda weights, dp_table: weights is not None, "Precondition failed: weights is not None")
@icontract.ensure(lambda result, weights, dp_table: selections.shape[0] == weights.shape[0], "Postcondition failed: selections.shape[0] == weights.shape[0]")
def backtrack_knapsack_items(weights: NDArray[np.int64], dp_table: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Walk back to resolve which specific indices are chosen.

    Args:
        weights: NDArray[np.int64]
        dp_table: NDArray[np.float64]

    Returns:
        selections: NDArray[np.bool_]
    """
    return needs_human_decision() # type: ignore


from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_run_hungarian_assignment,
)

@register_atom(witness_run_hungarian_assignment, name="run_hungarian_assignment")
@icontract.require(lambda cost_matrix: cost_matrix.ndim == 2, "Precondition failed: cost_matrix.ndim == 2")
@icontract.require(lambda cost_matrix: np.all(np.isfinite(cost_matrix)), "Precondition failed: np.all(np.isfinite(cost_matrix))")
@icontract.ensure(lambda result, cost_matrix: row_ind.shape[0] == min(cost_matrix.shape), "Postcondition failed: row_ind.shape[0] == min(cost_matrix.shape)")
@icontract.ensure(lambda result, cost_matrix: col_ind.shape[0] == row_ind.shape[0], "Postcondition failed: col_ind.shape[0] == row_ind.shape[0]")
def run_hungarian_assignment(cost_matrix: NDArray[np.float64]) -> NDArray[np.int64]:
    """Perform linear sum assignment optimization.

    Args:
        cost_matrix: NDArray[np.float64]

    Returns:
        row_ind: NDArray[np.int64]
    """
    import scipy.optimize
    return scipy.optimize.linear_sum_assignment(cost_matrix=cost_matrix) # type: ignore


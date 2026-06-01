from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_nnls,
)

@register_atom(witness_solve_nnls, name="solve_nnls")
@icontract.require(lambda A, b: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.require(lambda A, b: A.shape[0] == b.shape[0], "Precondition failed: A.shape[0] == b.shape[0]")
@icontract.ensure(lambda result, A, b: np.all(x >= 0.0), "Postcondition failed: np.all(x >= 0.0)")
def solve_nnls(A: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute nonnegative least squares via Lawson-Hanson active-set solver.

    Args:
        A: NDArray[np.float64]
        b: NDArray[np.float64]

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.optimize
    return scipy.optimize.nnls(A=A, b=b) # type: ignore


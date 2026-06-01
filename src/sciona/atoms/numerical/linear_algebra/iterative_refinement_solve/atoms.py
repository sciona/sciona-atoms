from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_calculate_high_prec_residual,
    witness_apply_refinement_step,
)

@register_atom(witness_calculate_high_prec_residual, name="calculate_high_prec_residual")
@icontract.require(lambda A, b, x: A.shape[0] == x.shape[0], "Precondition failed: A.shape[0] == x.shape[0]")
@icontract.ensure(lambda result, A, b, x: result is not None, "Postcondition failed: result is not None")
def calculate_high_prec_residual(A: NDArray[np.float64], b: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float128]:
    """Compute residual vector in double/quad precision.

    Args:
        A: NDArray[np.float64]
        b: NDArray[np.float64]
        x: NDArray[np.float64]

    Returns:
        r: NDArray[np.float128]
    """
    import numpy
    return numpy.dot(A=A, b=b, x=x) # type: ignore

@register_atom(witness_apply_refinement_step, name="apply_refinement_step")
@icontract.require(lambda lu_and_piv, r, x_old: lu_and_piv is not None, "Precondition failed: lu_and_piv is not None")
@icontract.ensure(lambda result, lu_and_piv, r, x_old: result is not None, "Postcondition failed: result is not None")
def apply_refinement_step(lu_and_piv: tuple[NDArray[np.float64], NDArray[np.int32]], r: NDArray[np.float128], x_old: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve for updates and apply to solution.

    Args:
        lu_and_piv: tuple[NDArray[np.float64], NDArray[np.int32]]
        r: NDArray[np.float128]
        x_old: NDArray[np.float64]

    Returns:
        x_new: NDArray[np.float64]
    """
    import scipy.linalg
    return scipy.linalg.lu_solve(lu_and_piv=lu_and_piv, r=r, x_old=x_old) # type: ignore


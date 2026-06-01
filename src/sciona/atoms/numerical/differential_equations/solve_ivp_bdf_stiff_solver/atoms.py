from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_bdf_predictor,
    witness_bdf_newton_corrector,
)

@register_atom(witness_bdf_predictor, name="bdf_predictor")
@icontract.require(lambda history, order: order >= 1, "Precondition failed: order >= 1")
@icontract.require(lambda history, order: order <= 5, "Precondition failed: order <= 5")
@icontract.ensure(lambda result, history, order: result is not None, "Postcondition failed: result is not None")
def bdf_predictor(history: NDArray[np.float64], order: int) -> NDArray[np.float64]:
    """Extrapolate previous states to calculate predictor values for BDF implicit equations.

    Args:
        history: NDArray[np.float64]
        order: int

    Returns:
        y_predict: NDArray[np.float64]
    """
    import scipy.integrate.BDF
    return scipy.integrate.BDF._predict(history=history, order=order) # type: ignore

@register_atom(witness_bdf_newton_corrector, name="bdf_newton_corrector")
@icontract.require(lambda fun, t_new, y_predict, h, beta, jacobian, rhs_constant, atol, rtol: fun is not None, "Precondition failed: fun is not None")
@icontract.ensure(lambda result, fun, t_new, y_predict, h, beta, jacobian, rhs_constant, atol, rtol: result is not None, "Postcondition failed: result is not None")
def bdf_newton_corrector(fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]], t_new: float, y_predict: NDArray[np.float64], h: float, beta: float, jacobian: NDArray[np.float64], rhs_constant: NDArray[np.float64], atol: float, rtol: float) -> NDArray[np.float64]:
    """Execute Newton iterations to solve the BDF implicit algebraic system.

    Args:
        fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
        t_new: float
        y_predict: NDArray[np.float64]
        h: float
        beta: float
        jacobian: NDArray[np.float64]
        rhs_constant: NDArray[np.float64]
        atol: float
        rtol: float

    Returns:
        y_corrected: NDArray[np.float64]
    """
    import scipy.integrate.BDF
    return scipy.integrate.BDF._solve_and_update(fun=fun, t_new=t_new, y_predict=y_predict, h=h, beta=beta, jacobian=jacobian, rhs_constant=rhs_constant, atol=atol, rtol=rtol) # type: ignore


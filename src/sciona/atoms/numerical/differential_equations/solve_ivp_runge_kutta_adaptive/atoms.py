from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_get_butcher_tableau,
    witness_compute_rk_single_step,
    witness_evaluate_step_acceptability,
)

@register_atom(witness_get_butcher_tableau, name="get_butcher_tableau")
@icontract.require(lambda method: method is not None, "Precondition failed: method is not None")
@icontract.ensure(lambda result, method: result is not None, "Postcondition failed: result is not None")
def get_butcher_tableau(method: str) -> NDArray[np.float64]:
    """Retrieve matrices A, coefficients B, B_star, and C defining an embedded Runge-Kutta scheme.

    Args:
        method: Must be 'RK45' or 'RK23'

    Returns:
        A: NDArray[np.float64]
    """
    import scipy.integrate
    return scipy.integrate.RK45(method=method) # type: ignore

@register_atom(witness_compute_rk_single_step, name="compute_rk_single_step")
@icontract.require(lambda fun, t, y, h, A, B, B_star, C: h > 0, "Precondition failed: h > 0")
@icontract.ensure(lambda result, fun, t, y, h, A, B, B_star, C: y_new.shape == y.shape, "Postcondition failed: y_new.shape == y.shape")
@icontract.ensure(lambda result, fun, t, y, h, A, B, B_star, C: error_vector.shape == y.shape, "Postcondition failed: error_vector.shape == y.shape")
def compute_rk_single_step(fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]], t: float, y: NDArray[np.float64], h: float, A: NDArray[np.float64], B: NDArray[np.float64], B_star: NDArray[np.float64], C: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate explicit Runge-Kutta stages and compute step candidate and error vector.

    Args:
        fun: Callable[[float, NDArray[np.float64]], NDArray[np.float64]]
        t: float
        y: NDArray[np.float64]
        h: float
        A: NDArray[np.float64]
        B: NDArray[np.float64]
        B_star: NDArray[np.float64]
        C: NDArray[np.float64]

    Returns:
        y_new: NDArray[np.float64]
    """
    import scipy.integrate.RK45
    return scipy.integrate.RK45._step_impl(fun=fun, t=t, y=y, h=h, A=A, B=B, B_star=B_star, C=C) # type: ignore

@register_atom(witness_evaluate_step_acceptability, name="evaluate_step_acceptability")
@icontract.require(lambda y, y_new, error_vector, atol, rtol: atol > 0, "Precondition failed: atol > 0")
@icontract.require(lambda y, y_new, error_vector, atol, rtol: rtol > 0, "Precondition failed: rtol > 0")
@icontract.ensure(lambda result, y, y_new, error_vector, atol, rtol: error_norm >= 0, "Postcondition failed: error_norm >= 0")
def evaluate_step_acceptability(y: NDArray[np.float64], y_new: NDArray[np.float64], error_vector: NDArray[np.float64], atol: float, rtol: float) -> bool:
    """Compare error norm against tolerances to decide step acceptance and compute next step scaling factor.

    Args:
        y: NDArray[np.float64]
        y_new: NDArray[np.float64]
        error_vector: NDArray[np.float64]
        atol: float
        rtol: float

    Returns:
        accepted: bool
    """
    import scipy.integrate.RK45
    return scipy.integrate.RK45._step_impl(y=y, y_new=y_new, error_vector=error_vector, atol=atol, rtol=rtol) # type: ignore


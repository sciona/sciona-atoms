from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_linear_operator,
    witness_verify_adjoint_relation,
)

@register_atom(witness_build_linear_operator, name="build_linear_operator")
@icontract.require(lambda shape, matvec, rmatvec: len(shape) == 2, "Precondition failed: len(shape) == 2")
@icontract.ensure(lambda result, shape, matvec, rmatvec: result is not None, "Postcondition failed: result is not None")
def build_linear_operator(shape: int, matvec: Callable[[NDArray[np.float64]], NDArray[np.float64]], rmatvec: Callable[[NDArray[np.float64]], NDArray[np.float64]] = None) -> Any:
    """Construct a custom LinearOperator wrapper around functions.

    Args:
        shape: tuple[int, int]
        matvec: Callable[[NDArray[np.float64]], NDArray[np.float64]]
        rmatvec: Callable[[NDArray[np.float64]], NDArray[np.float64]]

    Returns:
        operator: scipy.sparse.linalg.LinearOperator
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.LinearOperator(shape=shape, matvec=matvec, rmatvec=rmatvec) # type: ignore

@register_atom(witness_verify_adjoint_relation, name="verify_adjoint_relation")
@icontract.require(lambda operator: operator is not None, "Precondition failed: operator is not None")
@icontract.ensure(lambda result, operator: result is not None, "Postcondition failed: result is not None")
def verify_adjoint_relation(operator: Any) -> bool:
    """Perform stochastic test of the adjoint relation.

    Args:
        operator: scipy.sparse.linalg.LinearOperator

    Returns:
        is_valid: bool
    """
    pass


from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_kronecker_operator,
)

@register_atom(witness_build_kronecker_operator, name="build_kronecker_operator")
@icontract.require(lambda A, B: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.require(lambda A, B: B.ndim == 2, "Precondition failed: B.ndim == 2")
@icontract.ensure(lambda result, A, B: result is not None, "Postcondition failed: result is not None")
def build_kronecker_operator(A: Any, B: Any) -> Any:
    """Assemble a matrix-free Kronecker product LinearOperator.

    Args:
        A: scipy.sparse.linalg.LinearOperator
        B: scipy.sparse.linalg.LinearOperator

    Returns:
        kron_op: scipy.sparse.linalg.LinearOperator
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.LinearOperator(A=A, B=B) # type: ignore


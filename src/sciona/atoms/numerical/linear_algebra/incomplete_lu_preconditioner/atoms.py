from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_ilut_factors,
    witness_create_preconditioner_operator,
)

@register_atom(witness_compute_ilut_factors, name="compute_ilut_factors")
@icontract.require(lambda A, drop_tol: A.shape[0] == A.shape[1], "Precondition failed: A.shape[0] == A.shape[1]")
@icontract.ensure(lambda result, A, drop_tol: result is not None, "Postcondition failed: result is not None")
def compute_ilut_factors(A: Any, drop_tol: float = None) -> Any:
    """Compute incomplete LU factors with threshold-based dropping.

    Args:
        A: scipy.sparse.spmatrix
        drop_tol: float

    Returns:
        ilu_obj: scipy.sparse.linalg.SuperLU
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.spilu(A=A, drop_tol=drop_tol) # type: ignore

@register_atom(witness_create_preconditioner_operator, name="create_preconditioner_operator")
@icontract.require(lambda ilu_obj: ilu_obj is not None, "Precondition failed: ilu_obj is not None")
@icontract.ensure(lambda result, ilu_obj: result is not None, "Postcondition failed: result is not None")
def create_preconditioner_operator(ilu_obj: Any) -> Any:
    """Convert SuperLU factors into a LinearOperator.

    Args:
        ilu_obj: scipy.sparse.linalg.SuperLU

    Returns:
        M: scipy.sparse.linalg.LinearOperator
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.LinearOperator(ilu_obj=ilu_obj) # type: ignore


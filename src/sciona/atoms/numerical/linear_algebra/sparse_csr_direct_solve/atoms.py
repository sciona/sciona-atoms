from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_sparse_factorize_matrix,
    witness_sparse_solve_rhs,
)

@register_atom(witness_sparse_factorize_matrix, name="sparse_factorize_matrix")
@icontract.require(lambda A: A.shape[0] == A.shape[1], "Precondition failed: A.shape[0] == A.shape[1]")
@icontract.ensure(lambda result, A: result is not None, "Postcondition failed: result is not None")
def sparse_factorize_matrix(A: Any) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Perform sparse LU factorization via SuperLU.

    Args:
        A: scipy.sparse.spmatrix

    Returns:
        solve_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.factorized(A=A) # type: ignore

@register_atom(witness_sparse_solve_rhs, name="sparse_solve_rhs")
@icontract.require(lambda solve_fn, b: solve_fn is not None, "Precondition failed: solve_fn is not None")
@icontract.ensure(lambda result, solve_fn, b: x.shape == b.shape, "Postcondition failed: x.shape == b.shape")
def sparse_solve_rhs(solve_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate solution for right hand side.

    Args:
        solve_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]]
        b: NDArray[np.float64]

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.spsolve(solve_fn=solve_fn, b=b) # type: ignore


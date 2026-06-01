from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterate_conjugate_gradient,
)

@register_atom(witness_iterate_conjugate_gradient, name="iterate_conjugate_gradient")
@icontract.require(lambda A, b, M, tol, maxiter: b.ndim == 1, "Precondition failed: b.ndim == 1")
@icontract.ensure(lambda result, A, b, M, tol, maxiter: x.shape == b.shape, "Postcondition failed: x.shape == b.shape")
def iterate_conjugate_gradient(A: Any, b: NDArray[np.float64], M: Any = None, tol: float = None, maxiter: int = None) -> NDArray[np.float64]:
    """Run Conjugate Gradient iterations on A x = b.

    Args:
        A: scipy.sparse.linalg.LinearOperator
        b: NDArray[np.float64]
        M: scipy.sparse.linalg.LinearOperator
        tol: float
        maxiter: int

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.cg(A=A, b=b, M=M, tol=tol, maxiter=maxiter) # type: ignore


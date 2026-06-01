from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterate_bicgstab,
)

@register_atom(witness_iterate_bicgstab, name="iterate_bicgstab")
@icontract.require(lambda A, b, tol, maxiter: b.ndim == 1, "Precondition failed: b.ndim == 1")
@icontract.ensure(lambda result, A, b, tol, maxiter: x.shape == b.shape, "Postcondition failed: x.shape == b.shape")
def iterate_bicgstab(A: Any, b: NDArray[np.float64], tol: float = None, maxiter: int = None) -> NDArray[np.float64]:
    """Run stabilized bi-conjugate gradient steps.

    Args:
        A: scipy.sparse.linalg.LinearOperator
        b: NDArray[np.float64]
        tol: float
        maxiter: int

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.bicgstab(A=A, b=b, tol=tol, maxiter=maxiter) # type: ignore


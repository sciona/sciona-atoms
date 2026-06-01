from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_iterate_gmres,
)

@register_atom(witness_iterate_gmres, name="iterate_gmres")
@icontract.require(lambda A, b, restart, tol, maxiter: b.ndim == 1, "Precondition failed: b.ndim == 1")
@icontract.ensure(lambda result, A, b, restart, tol, maxiter: x.shape == b.shape, "Postcondition failed: x.shape == b.shape")
def iterate_gmres(A: Any, b: NDArray[np.float64], restart: int = None, tol: float = None, maxiter: int = None) -> NDArray[np.float64]:
    """Run restarted GMRES iterations on sparse linear system.

    Args:
        A: scipy.sparse.linalg.LinearOperator
        b: NDArray[np.float64]
        restart: int
        tol: float
        maxiter: int

    Returns:
        x: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.gmres(A=A, b=b, restart=restart, tol=tol, maxiter=maxiter) # type: ignore


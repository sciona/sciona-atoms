from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_qr_factorize,
    witness_qr_solve_least_squares,
)

@register_atom(witness_qr_factorize, name="qr_factorize")
@icontract.require(lambda A, pivoting: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.ensure(lambda result, A, pivoting: q.shape[0] == A.shape[0], "Postcondition failed: q.shape[0] == A.shape[0]")
@icontract.ensure(lambda result, A, pivoting: r.shape[1] == A.shape[1], "Postcondition failed: r.shape[1] == A.shape[1]")
def qr_factorize(A: NDArray[np.float64 | np.complex128], pivoting: bool = None) -> NDArray[np.float64 | np.complex128]:
    """Compute QR decomposition with column pivoting.

    Args:
        A: 2D dense matrix
        pivoting: Default True

    Returns:
        q: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.qr(A=A, pivoting=pivoting) # type: ignore

@register_atom(witness_qr_solve_least_squares, name="qr_solve_least_squares")
@icontract.require(lambda q, r, b, p: b.shape[0] == q.shape[0], "Precondition failed: b.shape[0] == q.shape[0]")
@icontract.ensure(lambda result, q, r, b, p: x.shape[0] == r.shape[1], "Postcondition failed: x.shape[0] == r.shape[1]")
def qr_solve_least_squares(q: NDArray[np.float64 | np.complex128], r: NDArray[np.float64 | np.complex128], b: NDArray[np.float64 | np.complex128], p: NDArray[np.int32] = None) -> NDArray[np.float64 | np.complex128]:
    """Solve least squares using precomputed QR.

    Args:
        q: NDArray[np.float64 | np.complex128]
        r: NDArray[np.float64 | np.complex128]
        b: NDArray[np.float64 | np.complex128]
        p: NDArray[np.int32]

    Returns:
        x: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.lstsq(q=q, r=r, b=b, p=p) # type: ignore


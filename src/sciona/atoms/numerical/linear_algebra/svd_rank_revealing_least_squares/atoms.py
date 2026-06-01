from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_svd_decompose,
    witness_svd_threshold_solve,
)

@register_atom(witness_svd_decompose, name="svd_decompose")
@icontract.require(lambda A: A.ndim == 2, "Precondition failed: A.ndim == 2")
@icontract.ensure(lambda result, A: s.ndim == 1, "Postcondition failed: s.ndim == 1")
def svd_decompose(A: NDArray[np.float64 | np.complex128]) -> NDArray[np.float64 | np.complex128]:
    """Compute complete SVD of A.

    Args:
        A: NDArray[np.float64 | np.complex128]

    Returns:
        U: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.svd(A=A) # type: ignore

@register_atom(witness_svd_threshold_solve, name="svd_threshold_solve")
@icontract.require(lambda U, s, Vh, b, cond: b.shape[0] == U.shape[0], "Precondition failed: b.shape[0] == U.shape[0]")
@icontract.ensure(lambda result, U, s, Vh, b, cond: x.shape[0] == Vh.shape[1], "Postcondition failed: x.shape[0] == Vh.shape[1]")
def svd_threshold_solve(U: NDArray[np.float64 | np.complex128], s: NDArray[np.float64], Vh: NDArray[np.float64 | np.complex128], b: NDArray[np.float64 | np.complex128], cond: float = None) -> NDArray[np.float64 | np.complex128]:
    """Apply singular value thresholding and solve for x.

    Args:
        U: NDArray[np.float64 | np.complex128]
        s: NDArray[np.float64]
        Vh: NDArray[np.float64 | np.complex128]
        b: NDArray[np.float64 | np.complex128]
        cond: Threshold below which singular values are zeroed

    Returns:
        x: NDArray[np.float64 | np.complex128]
    """
    import scipy.linalg
    return scipy.linalg.lstsq(U=U, s=s, Vh=Vh, b=b, cond=cond) # type: ignore


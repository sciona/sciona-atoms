from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_gram_matrix,
    witness_apply_tikhonov_shift_and_solve,
)

@register_atom(witness_compute_gram_matrix, name="compute_gram_matrix")
@icontract.require(lambda A, b: A.ndim == 2)
@icontract.require(lambda A, b: A.shape[0] == b.shape[0])
@icontract.ensure(lambda result: result[0].shape[0] == result[0].shape[1])
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])))
@icontract.ensure(lambda result: np.all(np.isfinite(result[1])))
def compute_gram_matrix(
    A: NDArray[np.float64],
    b: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Assemble Gram matrix and projected RHS.

    Parameters
    ----------
    A : NDArray[np.float64]
        Coefficient matrix of shape (m, n).
    b : NDArray[np.float64]
        Right-hand side vector/matrix of shape (m,) or (m, k).

    Returns
    -------
    Gram : NDArray[np.float64]
        Gram matrix A.T @ A of shape (n, n).
    Ab : NDArray[np.float64]
        Projected right-hand side A.T @ b of shape (n,) or (n, k).
    """
    Gram = A.T @ A
    Ab = A.T @ b
    return Gram, Ab

@register_atom(witness_apply_tikhonov_shift_and_solve, name="apply_tikhonov_shift_and_solve")
@icontract.require(lambda Gram, Ab, alpha: alpha >= 0.0)
@icontract.require(lambda Gram, Ab, alpha: Gram.ndim == 2)
@icontract.require(lambda Gram, Ab, alpha: Gram.shape[0] == Gram.shape[1])
@icontract.require(lambda Gram, Ab, alpha: Gram.shape[0] == Ab.shape[0])
@icontract.ensure(lambda result, Ab: result.shape == Ab.shape)
@icontract.ensure(lambda result: np.all(np.isfinite(result)))
def apply_tikhonov_shift_and_solve(
    Gram: NDArray[np.float64],
    Ab: NDArray[np.float64],
    alpha: float,
) -> NDArray[np.float64]:
    """Apply diagonal regularization shift and solve via Cholesky.

    Parameters
    ----------
    Gram : NDArray[np.float64]
        Gram matrix of shape (n, n).
    Ab : NDArray[np.float64]
        Projected right-hand side of shape (n,) or (n, k).
    alpha : float
        Regularization parameter (alpha >= 0).

    Returns
    -------
    x : NDArray[np.float64]
        Regularized least squares solution.
    """
    import scipy.linalg
    shifted_Gram = Gram + alpha * np.eye(Gram.shape[0])
    c, low = scipy.linalg.cho_factor(shifted_Gram, lower=False)
    x = scipy.linalg.cho_solve((c, low), Ab)
    return x



"""Symmetric Positive Definite Cholesky Solver family.

Provides high-performance, stable direct square solvers for symmetric (or
Hermitian) positive-definite linear systems A x = b. Using Cholesky
factorization (A = L L^H or U^H U) saves exactly half of the flops
required by general dense LU solvers (O(N^3/3) vs O(2N^3/3) flops).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.linalg  # type: ignore[import-untyped]
import icontract

from sciona.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import (
    witness_cholesky_factorize,
    witness_cholesky_solve,
)


@register_atom(witness_cholesky_factorize)  # type: ignore[untyped-decorator]
@icontract.require(lambda A: A.ndim == 2)
@icontract.require(lambda A: A.shape[0] == A.shape[1])
@icontract.require(lambda A: A.shape[0] > 0)
@icontract.require(lambda A: np.allclose(A, A.T.conj()))
@icontract.ensure(lambda result: len(result) == 2)
@icontract.ensure(lambda result, A: result[0].shape == A.shape)
def cholesky_factorize(
    A: NDArray[np.float64] | NDArray[np.complex128],
    lower: bool = False,
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], bool]:
    """Compute the Cholesky factorization of a symmetric/Hermitian positive-definite matrix A.

    Decomposes the coefficient matrix A such that A = U^H U (if lower=False)
    or L L^H (if lower=True).

    Parameters
    ----------
    A : NDArray[np.float64] or NDArray[np.complex128]
        Symmetric/Hermitian positive-definite dense matrix of shape (n, n).
    lower : bool, optional
        Whether to compute the lower triangular factor L (True) or upper
        triangular factor U (False) (default False).

    Returns
    -------
    c : NDArray[np.float64] or NDArray[np.complex128]
        The Cholesky factor matrix (upper or lower triangular, same shape as A).
    lower : bool
        Flag indicating if the factorization represents lower or upper storage.
    """
    c, low = scipy.linalg.cho_factor(A, lower=lower)
    return c, low


@register_atom(witness_cholesky_solve)  # type: ignore[untyped-decorator]
@icontract.require(lambda c_factor: len(c_factor) == 2)
@icontract.require(lambda c_factor: c_factor[0].ndim == 2)
@icontract.require(lambda c_factor: c_factor[0].shape[0] == c_factor[0].shape[1])
@icontract.require(lambda c_factor, b: b.shape[0] == c_factor[0].shape[0])
@icontract.ensure(lambda result, b: result.shape == b.shape)
def cholesky_solve(
    c_factor: tuple[NDArray[np.float64] | NDArray[np.complex128], bool],
    b: NDArray[np.float64] | NDArray[np.complex128],
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Solve the symmetric positive-definite system A x = b using Cholesky factor.

    Uses back-substitution solvers given the precomputed Cholesky factor representation.

    Parameters
    ----------
    c_factor : tuple[NDArray, bool]
        The Cholesky factor matrix and lower/upper flag as returned by
        cholesky_factorize.
    b : NDArray[np.float64] or NDArray[np.complex128]
        Right-hand side vector or matrix of shape (n,) or (n, k).

    Returns
    -------
    x : NDArray[np.float64] or NDArray[np.complex128]
        Solution vector or matrix satisfying A x = b, matching the shape of b.
    """
    x = scipy.linalg.cho_solve(c_factor, b)
    return x  # type: ignore[no-any-return]

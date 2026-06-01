"""Dense Pivoted LU Linear Solver family.

Provides high-performance, stable direct squares solvers using LU matrix
decomposition with partial row pivoting. Decomposing the system solves
numerical stability concerns and facilitates solving against multiple
right-hand sides.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import scipy.linalg  # type: ignore[import-untyped]
import scipy.linalg.lapack  # type: ignore[import-untyped]
import icontract

from sciona.ghost.registry import register_atom  # type: ignore[import-untyped]
from .witnesses import (
    witness_estimate_lu_condition_number,
    witness_lu_factorize_matrix,
    witness_lu_solve_system,
)


@register_atom(witness_lu_factorize_matrix)  # type: ignore[untyped-decorator]
@icontract.require(lambda A: A.ndim == 2)
@icontract.require(lambda A: A.shape[0] == A.shape[1])
@icontract.require(lambda A: A.shape[0] > 0)
@icontract.ensure(lambda result: len(result) == 2)
@icontract.ensure(lambda result, A: result[0].shape == A.shape)
@icontract.ensure(lambda result, A: result[1].shape == (A.shape[0],))
def lu_factorize_matrix(
    A: NDArray[np.float64] | NDArray[np.complex128],
) -> tuple[NDArray[np.float64] | NDArray[np.complex128], NDArray[np.int32]]:
    """Compute the LU factorization of a general square matrix A.

    Decomposes the coefficient matrix A such that P A = L U using partial
    pivoting with row interchanges.

    Parameters
    ----------
    A : NDArray[np.float64] or NDArray[np.complex128]
        Square dense matrix of shape (n, n) to be decomposed.

    Returns
    -------
    lu : NDArray[np.float64] or NDArray[np.complex128]
        A matrix containing U in the upper triangle and L (excluding diagonal)
        in the lower triangle.
    piv : NDArray[np.int32]
        The pivot indices that define the permutation matrix P.
    """
    lu, piv = scipy.linalg.lu_factor(A)
    return lu, piv


@register_atom(witness_lu_solve_system)  # type: ignore[untyped-decorator]
@icontract.require(lambda lu_and_piv: len(lu_and_piv) == 2)
@icontract.require(lambda lu_and_piv: lu_and_piv[0].ndim == 2)
@icontract.require(lambda lu_and_piv: lu_and_piv[0].shape[0] == lu_and_piv[0].shape[1])
@icontract.require(lambda lu_and_piv, b: b.shape[0] == lu_and_piv[0].shape[0])
@icontract.ensure(lambda result, b: result.shape == b.shape)
def lu_solve_system(
    lu_and_piv: tuple[NDArray[np.float64] | NDArray[np.complex128], NDArray[np.int32]],
    b: NDArray[np.float64] | NDArray[np.complex128],
) -> NDArray[np.float64] | NDArray[np.complex128]:
    """Solve a square linear system A x = b using LU factorization.

    Uses backward and forward substitution to solve the pivoted system.

    Parameters
    ----------
    lu_and_piv : tuple of (NDArray, NDArray)
        Packed LU factorization and permutation pivot vector as returned by
        lu_factorize_matrix.
    b : NDArray[np.float64] or NDArray[np.complex128]
        Right-hand side vector or matrix of shape (n,) or (n, k).

    Returns
    -------
    x : NDArray[np.float64] or NDArray[np.complex128]
        Solution vector or matrix satisfying A x = b, matching the shape of b.
    """
    x = scipy.linalg.lu_solve(lu_and_piv, b)
    return x  # type: ignore[no-any-return]


@register_atom(witness_estimate_lu_condition_number)  # type: ignore[untyped-decorator]
@icontract.require(lambda lu_and_piv: len(lu_and_piv) == 2)
@icontract.require(lambda lu_and_piv: lu_and_piv[0].ndim == 2)
@icontract.require(lambda lu_and_piv: lu_and_piv[0].shape[0] == lu_and_piv[0].shape[1])
@icontract.require(lambda norm_A: norm_A >= 0.0)
@icontract.ensure(lambda result: 0.0 <= result <= 1.0)
def estimate_lu_condition_number(
    lu_and_piv: tuple[NDArray[np.float64] | NDArray[np.complex128], NDArray[np.int32]],
    norm_A: float,
) -> float:
    """Estimate the reciprocal condition number of A based on LU factors.

    Uses LAPACK's dgecon (for real float arrays) or zgecon (for complex float arrays).

    Parameters
    ----------
    lu_and_piv : tuple of (NDArray, NDArray)
        The factored matrix LU and the pivot indices as returned by lu_factorize_matrix.
    norm_A : float
        The 1-norm or infinity norm of the original matrix A.

    Returns
    -------
    rcond : float
        The estimated reciprocal condition number (rcond).
    """
    lu, _ = lu_and_piv
    if np.iscomplexobj(lu):
        lu_complex = lu.astype(np.complex128, copy=False)
        rcond, info = scipy.linalg.lapack.zgecon(lu_complex, norm_A)
    else:
        lu_real = lu.astype(np.float64, copy=False)
        rcond, info = scipy.linalg.lapack.dgecon(lu_real, norm_A)

    if info != 0:
        raise ValueError(f"LAPACK condition number estimation failed with info={info}")

    return float(rcond)

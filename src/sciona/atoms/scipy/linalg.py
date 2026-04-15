from __future__ import annotations
from typing import Any, Tuple, Union
import numpy as np
import scipy.linalg
import icontract
from sciona.ghost.registry import register_atom
from sciona.atoms.scipy.witnesses import (
    witness_scipy_linalg_det,
    witness_scipy_linalg_inv,
    witness_scipy_linalg_solve,
    witness_scipy_lu_factor,
    witness_scipy_lu_solve,
)

# Types
ArrayLike = Union[np.ndarray, list, tuple]

def _is_square_2d(a: ArrayLike) -> bool:
    """Check that a is a 2D square matrix."""
    a_arr = np.asarray(a)
    return a_arr.ndim == 2 and a_arr.shape[0] == a_arr.shape[1]

@register_atom(witness_scipy_linalg_solve, name="scipy.linalg.solve")
@icontract.require(lambda a, b: np.asarray(a).ndim == 2, "a must be a 2D matrix")
@icontract.require(lambda a, b: _is_square_2d(a), "a must be square")
@icontract.require(lambda a, b: np.asarray(a).shape[0] == np.asarray(b).shape[0], "Dimensions of a and b must match")
@icontract.ensure(lambda result, a, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def solve(
    a: ArrayLike,
    b: ArrayLike,
    lower: bool = False,
    overwrite_a: bool = False,
    overwrite_b: bool = False,
    check_finite: bool = True,
    assume_a: str | None = None,
    transposed: bool = False,
) -> np.ndarray:
    """Solve the linear equation a @ x == b for x.

    Args:
        a: Coefficient matrix, shape (n, n).
        b: Ordinate values, shape (n,) or (n, k).
        lower: Use only data contained in the lower triangle of a.
            Default is to use upper triangle.
        overwrite_a: Allow overwriting data in a (may enhance
            performance).
        overwrite_b: Allow overwriting data in b (may enhance
            performance).
        check_finite: Whether to check that the input matrices contain
            only finite numbers.
        assume_a: Type of data in a. Default is 'gen' (general).

    Returns:
        Solution to the system a @ x == b. Shape matches b.

    """
    return scipy.linalg.solve(
        a,
        b,
        lower=lower,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
        assume_a=assume_a,
        transposed=transposed,
    )

@register_atom(witness_scipy_linalg_inv, name="scipy.linalg.inv")
@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result.shape == np.asarray(a).shape, "Inverse has same shape as input")
def inv(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """Compute the inverse of a matrix.

    Args:
        a: Square matrix to be inverted.
        overwrite_a: Allow overwriting data in a (may enhance
            performance).
        check_finite: Whether to check that the input matrix contains
            only finite numbers.

    Returns:
        Inverse of the matrix a.

    """
    return scipy.linalg.inv(a, overwrite_a=overwrite_a, check_finite=check_finite)

@register_atom(witness_scipy_linalg_det, name="scipy.linalg.det")
@icontract.require(lambda a: np.asarray(a).ndim >= 2, "a must have at least 2 dimensions")
@icontract.require(lambda a: np.asarray(a).shape[-1] == np.asarray(a).shape[-2], "Last two dimensions of a must be square")
@icontract.ensure(lambda result: result is not None, "Determinant must not be None")
def det(a: ArrayLike, overwrite_a: bool = False, check_finite: bool = True) -> float:
    """Compute the determinant of a matrix.

    Args:
        a: Square matrix, shape (..., M, M), of which the determinant
            is computed.
        overwrite_a: Allow overwriting data in a (may enhance
            performance).
        check_finite: Whether to check that the input matrix contains
            only finite numbers.

    Returns:
        Determinant of a.

    """
    return float(scipy.linalg.det(a, overwrite_a=overwrite_a, check_finite=check_finite))

@register_atom(witness_scipy_lu_factor, name="scipy.linalg.lu_factor")
@icontract.require(lambda a: _is_square_2d(a), "a must be a square 2D matrix")
@icontract.ensure(lambda result, a: result[0].shape == np.asarray(a).shape, "LU factor has same shape as input")
@icontract.ensure(lambda result, a: result[1].shape == (np.asarray(a).shape[0],), "Pivot array has length n")
def lu_factor(
    a: ArrayLike,
    overwrite_a: bool = False,
    check_finite: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pivoted Lower-Upper triangular decomposition (LU) of a square matrix.

    The decomposition satisfies A = P @ L @ U where P is a permutation
    matrix derived from the pivot indices.

    Args:
        a: Square matrix to decompose, shape (n, n).
        overwrite_a: Whether to overwrite data in a (may improve
            performance).
        check_finite: Whether to check that the input contains only
            finite numbers.

    Returns:
        Tuple of (lu, piv) where lu is the LU factor matrix of shape
        (n, n) and piv is the pivot index array of shape (n,).

    """
    return scipy.linalg.lu_factor(a, overwrite_a=overwrite_a, check_finite=check_finite)

@register_atom(witness_scipy_lu_solve, name="scipy.linalg.lu_solve")
@icontract.require(lambda lu_and_piv, b: len(lu_and_piv) == 2, "lu_and_piv must be a tuple of (lu, piv)")
@icontract.require(lambda lu_and_piv, b: lu_and_piv[0].shape[0] == np.asarray(b).shape[0], "Dimensions of LU and b must match")
@icontract.ensure(lambda result, b: result.shape == np.asarray(b).shape, "Result shape must match b shape")
def lu_solve(
    lu_and_piv: Tuple[np.ndarray, np.ndarray],
    b: ArrayLike,
    trans: int = 0,
    overwrite_b: bool = False,
    check_finite: bool = True,
) -> np.ndarray:
    """Solve an equation system, a @ x = b, given the Lower-Upper triangular decomposition (LU) factorization of a.

    Args:
        lu_and_piv: Factorization of the coefficient matrix a, as given
            by lu_factor.
        b: Right-hand side.
        trans: Type of system to solve: 0: a @ x = b, 1: a^T @ x = b,
            2: a^H @ x = b.
        overwrite_b: Whether to overwrite data in b (may improve
            performance).
        check_finite: Whether to check that the input contains only
            finite numbers.

    Returns:
        Solution to the system a @ x = b.

    """
    return scipy.linalg.lu_solve(
        lu_and_piv,
        b,
        trans=trans,
        overwrite_b=overwrite_b,
        check_finite=check_finite,
    )

"""Ghost witnesses for the dense pivoted LU linear solver family.

Witnesses propagate matrix shape and dtype metadata through abstract types
without running any actual numeric matrix calculations.
"""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray  # type: ignore[import-untyped]


def witness_lu_factorize_matrix(
    A: AbstractArray,
) -> tuple[AbstractArray, AbstractArray]:
    """Ghost witness for LU factorization with partial row pivoting.

    Decomposes a square 2D array A into a factored representation
    comprising a packed LU matrix and a pivot indices array.

    Parameters
    ----------
    A : AbstractArray
        A square 2D matrix of shape (n, n).

    Returns
    -------
    tuple[AbstractArray, AbstractArray]
        A tuple of (LU matrix of shape (n, n), pivot array of shape (n,)).
    """
    n = A.shape[0] if len(A.shape) > 0 else 1
    return (
        AbstractArray(shape=(n, n), dtype=A.dtype),
        AbstractArray(shape=(n,), dtype="int32"),
    )


def witness_lu_solve_system(
    lu_and_piv: tuple[AbstractArray, AbstractArray],
    b: AbstractArray,
) -> AbstractArray:
    """Ghost witness for back-solving an LU-factored linear system.

    Computes the solution vector or matrix x that satisfies LU x = b.

    Parameters
    ----------
    lu_and_piv : tuple[AbstractArray, AbstractArray]
        The packed LU matrix of shape (n, n) and pivot array of shape (n,).
    b : AbstractArray
        The right-hand side vector/matrix of shape (n,) or (n, k).

    Returns
    -------
    AbstractArray
        The solution matrix/vector of the same shape and dtype as b.
    """
    return AbstractArray(shape=b.shape, dtype=b.dtype)


def witness_estimate_lu_condition_number(
    lu_and_piv: tuple[AbstractArray, AbstractArray],
    norm_A: float,
) -> float:
    """Ghost witness for reciprocal condition number estimation.

    Estimates the 1-norm or infinity-norm reciprocal condition number.

    Parameters
    ----------
    lu_and_piv : tuple[AbstractArray, AbstractArray]
        The packed LU matrix of shape (n, n) and pivot array of shape (n,).
    norm_A : float
        The norm of the original matrix A.

    Returns
    -------
    float
        The estimated reciprocal condition number.
    """
    return 0.0

"""Ghost witnesses for the symmetric positive definite Cholesky solver family.

Witnesses propagate matrix shape and dtype metadata through abstract types
without running any actual numeric matrix calculations.
"""

from __future__ import annotations

from sciona.ghost.abstract import AbstractArray  # type: ignore[import-untyped]


def witness_cholesky_factorize(
    A: AbstractArray,
    lower: bool = False,
) -> tuple[AbstractArray, bool]:
    """Ghost witness for Cholesky factorization of an SPD matrix.

    Decomposes a symmetric positive definite matrix A into L L^T or U^T U.

    Parameters
    ----------
    A : AbstractArray
        A square symmetric positive definite 2D matrix.
    lower : bool, optional
        Flag indicating lower or upper triangular storage (default False).

    Returns
    -------
    tuple[AbstractArray, bool]
        A tuple of (Cholesky factor matrix, lower flag).
    """
    return (AbstractArray(shape=A.shape, dtype=A.dtype), lower)


def witness_cholesky_solve(
    c_factor: tuple[AbstractArray, bool],
    b: AbstractArray,
) -> AbstractArray:
    """Ghost witness for solving SPD systems using Cholesky factor.

    Computes the solution vector or matrix x that satisfies A x = b.

    Parameters
    ----------
    c_factor : tuple[AbstractArray, bool]
        The Cholesky factor matrix and lower/upper flag.
    b : AbstractArray
        The right-hand side vector/matrix of shape (n,) or (n, k).

    Returns
    -------
    AbstractArray
        The solution matrix/vector of the same shape and dtype as b.
    """
    return AbstractArray(shape=b.shape, dtype=b.dtype)

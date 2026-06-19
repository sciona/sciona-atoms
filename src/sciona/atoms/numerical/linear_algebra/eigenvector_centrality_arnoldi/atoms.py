from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_solve_dominant_eigenvector,
    witness_perron_frobenius_correct,
)

@register_atom(witness_solve_dominant_eigenvector, name="solve_dominant_eigenvector")
@icontract.require(lambda adj_matrix, max_iter: max_iter > 0)
@icontract.require(lambda adj_matrix, max_iter: adj_matrix.shape[0] == adj_matrix.shape[1])
@icontract.ensure(lambda result, adj_matrix: result[1].shape == (adj_matrix.shape[0],))
@icontract.ensure(lambda result: np.all(np.isfinite(result[1])))
def solve_dominant_eigenvector(
    adj_matrix: scipy.sparse.spmatrix,
    max_iter: int,
) -> Tuple[complex, NDArray[np.complex128]]:
    """Call ARPACK to extract the largest eigenvalue/vector.

    Parameters
    ----------
    adj_matrix : scipy.sparse.spmatrix
        Adjacency matrix of shape (n, n).
    max_iter : int
        Maximum number of Arnoldi update iterations.

    Returns
    -------
    eigenvalue : complex
        Largest magnitude eigenvalue.
    eigenvector : NDArray[np.complex128]
        Corresponding eigenvector of shape (n,).
    """
    import scipy.sparse.linalg
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigs(
        adj_matrix, k=1, which="LM", maxiter=max_iter
    )
    eigenvalue = eigenvalues[0]
    eigenvector = eigenvectors[:, 0]
    return eigenvalue, eigenvector

@register_atom(witness_perron_frobenius_correct, name="perron_frobenius_correct")
@icontract.require(lambda eigenvector: eigenvector.ndim == 1)
@icontract.ensure(lambda result, eigenvector: result.shape == eigenvector.shape)
@icontract.ensure(lambda result: np.all(result >= 0.0))
@icontract.ensure(lambda result: np.all(np.isfinite(result)))
def perron_frobenius_correct(
    eigenvector: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Cast complex values, verify positiveness, and normalize vector.

    Parameters
    ----------
    eigenvector : NDArray[np.complex128]
        Complex eigenvector of shape (n,).

    Returns
    -------
    scores : NDArray[np.float64]
        Corrected and normalized real scores of shape (n,).
    """
    r = np.real(eigenvector)
    if np.sum(r) < 0:
        r = -r
    r = np.maximum(r, 0.0)
    nrm = np.linalg.norm(r)
    if nrm > 0:
        r = r / nrm
    return r



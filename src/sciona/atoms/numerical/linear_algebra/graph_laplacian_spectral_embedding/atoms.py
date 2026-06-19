from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_compute_laplacian_matrix,
    witness_solve_smallest_eigen,
)

@register_atom(witness_compute_laplacian_matrix, name="compute_laplacian_matrix")
@icontract.require(lambda adj_matrix, normalized: adj_matrix.shape[0] == adj_matrix.shape[1])
@icontract.ensure(lambda result, adj_matrix: result.shape == adj_matrix.shape)
def compute_laplacian_matrix(
    adj_matrix: scipy.sparse.spmatrix,
    normalized: bool,
) -> scipy.sparse.spmatrix:
    """Build graph Laplacian L = D - A.

    Parameters
    ----------
    adj_matrix : scipy.sparse.spmatrix
        Adjacency matrix of shape (n, n).
    normalized : bool
        Whether to compute normalized Laplacian.

    Returns
    -------
    laplacian : scipy.sparse.spmatrix
        Graph Laplacian matrix of shape (n, n).
    """
    import scipy.sparse.csgraph
    # csgraph.laplacian expects 'normed' argument for normalization
    return scipy.sparse.csgraph.laplacian(adj_matrix, normed=normalized)

@register_atom(witness_solve_smallest_eigen, name="solve_smallest_eigen")
@icontract.require(lambda laplacian, k: k < laplacian.shape[0])
@icontract.require(lambda laplacian, k: k > 0)
@icontract.ensure(lambda result, k: result[0].shape == (k,))
@icontract.ensure(lambda result, k, laplacian: result[1].shape == (laplacian.shape[0], k))
@icontract.ensure(lambda result: np.all(np.isfinite(result[0])))
@icontract.ensure(lambda result: np.all(np.isfinite(result[1])))
def solve_smallest_eigen(
    laplacian: scipy.sparse.spmatrix,
    k: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Extract d+1 smallest eigenvalues/eigenvectors using sparse solvers.

    Parameters
    ----------
    laplacian : scipy.sparse.spmatrix
        Graph Laplacian of shape (n, n).
    k : int
        Number of eigenvalues/eigenvectors to extract.

    Returns
    -------
    eigenvalues : NDArray[np.float64]
        Smallest eigenvalues of shape (k,).
    eigenvectors : NDArray[np.float64]
        Corresponding eigenvectors of shape (n, k).
    """
    import scipy.sparse.linalg
    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(laplacian, k=k, which='SM')
    return eigenvalues, eigenvectors



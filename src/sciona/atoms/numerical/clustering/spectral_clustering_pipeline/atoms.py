from __future__ import annotations

from typing import Any, Tuple, Union, List, Dict, Optional
import numpy as np
from numpy.typing import NDArray
import icontract
from sciona.ghost.registry import register_atom

from .witnesses import (
    witness_build_graph_laplacian,
    witness_solve_laplacian_eigenvectors,
)

@register_atom(witness_build_graph_laplacian, name="build_graph_laplacian")
@icontract.require(lambda affinity: affinity.shape[0] == affinity.shape[1], "Precondition failed: affinity.shape[0] == affinity.shape[1]")
@icontract.ensure(lambda result, affinity: laplacian.shape == affinity.shape, "Postcondition failed: laplacian.shape == affinity.shape")
def build_graph_laplacian(affinity: Any) -> Any:
    """Construct a normalized symmetric Graph Laplacian from an affinity similarity matrix.

    Args:
        affinity: scipy.sparse.csr_matrix

    Returns:
        laplacian: scipy.sparse.csr_matrix
    """
    import scipy.sparse.csgraph
    return scipy.sparse.csgraph.laplacian(affinity=affinity) # type: ignore

@register_atom(witness_solve_laplacian_eigenvectors, name="solve_laplacian_eigenvectors")
@icontract.require(lambda laplacian, n_components: n_components < laplacian.shape[0], "Precondition failed: n_components < laplacian.shape[0]")
@icontract.ensure(lambda result, laplacian, n_components: eigenvectors.shape == (laplacian.shape[0], n_components), "Postcondition failed: eigenvectors.shape == (laplacian.shape[0], n_components)")
def solve_laplacian_eigenvectors(laplacian: Any, n_components: int) -> NDArray[np.float64]:
    """Solve the generalized eigenvalue problem for the Graph Laplacian to retrieve low-dimensional representations.

    Args:
        laplacian: scipy.sparse.csr_matrix
        n_components: int

    Returns:
        eigenvalues: NDArray[np.float64]
    """
    import scipy.sparse.linalg
    return scipy.sparse.linalg.eigsh(laplacian=laplacian, n_components=n_components) # type: ignore

